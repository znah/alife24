const tapeLen=128, tapeN = 8*1024;

const $ = q=>document.querySelector(q);
const nextFrame = async()=> new Promise(requestAnimationFrame);

let wgpu, data;
let canvas, context, screenGroup

async function main() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice({requiredFeatures: ["bgra8unorm-storage"]});
    if (!device) {
        alert('need a browser that supports WebGPU');
        return;
    }
    
    canvas = $('canvas');
    context = canvas.getContext('webgpu');
    const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({device, format: preferredFormat, usage:GPUTextureUsage.STORAGE_BINDING});

    wgpu = new WGPU(device, /*wgsl*/`
        struct Grid{
            @builtin(global_invocation_id) id: vec3u,
            @builtin(num_workgroups) size: vec3u
        };

        // https://github.com/skeeto/hash-prospector
        fn hash(y:u32) -> u32{
            var x = y; // +1
            x ^= x >> 17;
            x *= 0xed5ad4bb;
            x ^= x >> 11;
            x *= 0xac4c1b51;
            x ^= x >> 15;
            x *= 0x31848bab;
            x ^= x >> 14;
            return x;
        }        
    `);

    data = wgpu.group(/*wgsl*/`
        const TapeSize = 128;
        const ChunkSize = TapeSize/2;
        struct Params {seed: u32, shuffleStride: u32};
        struct Counters {step_n:u32, op_n:u32};

        var<uniform> params: Params;
        var<storage> tapes: array<u32>;
        var<storage> stat: array<Counters>;
        var<storage> shuffleIdx: array<u32>;
        var<storage,read> font: array<u32>;
    `).make({
        tapes: new Uint8Array(tapeLen * tapeN),  
        stat: new Uint32Array(tapeN*2),
        shuffleIdx: new Uint32Array(tapeN*2),
        params: new Uint32Array([42+1, 1]), // seed, shuffleStride // transition aroung 9500 epoch
        font:FONT_8X16});

    wgpu.compute(tapeLen * tapeN / 4, 64, data, /*wgsl*/`
    fn main(grid:Grid) {
        let i = grid.id.x;
        tapes[i] = hash(params.seed+hash(i));
        if (i<arrayLength(&shuffleIdx)) {
            shuffleIdx[i] = i;
        }
    }`);

    screenGroup = wgpu.group(/*wgsl*/`
        var screen: texture_storage_2d<${preferredFormat},write>;
    `);
    render();
    
    let epochCount = 0;
    //const dirHandle= await showDirectoryPicker({id:'dump', readwrite:'readwrite'});

    async function dumpArray(name) {
        const buf = data[name];
        const blob = new Blob([buf], {type: 'application/octet-stream'});
        const fn = `${(epochCount).toString().padStart(6,0)}_${name}.dat`;
        const f = await dirHandle.getFileHandle(fn, {create:true});
        const wr = await f.createWritable();
        await wr.write(blob);
        await wr.close();
    }
    async function dumpArrays() {
        await data.tapes.pull();
        await data.shuffleIdx.pull();
        await data.stat.pull();
        await Promise.all(['params', 'tapes', 'shuffleIdx', 'stat'].map(dumpArray));
    }

    let statTime = 0;
    let stepsInQueue = 0, reactCount = 0;
    let MstepSec = 0, MopSec = 0;
    while(1) { //epochCount < 12000
        while (stepsInQueue < 8) {
            step(); stepsInQueue++;
            device.queue.onSubmittedWorkDone().then(()=>{
                reactCount += tapeN;
                epochCount += 1;
                stepsInQueue--;
            });
        }
        render();
        const t = Date.now();
        if (t-statTime > 1000) {
            const prevStatTime = statTime;
            data.stat.pull().then(stat=>{
                let stepCount=0, opCount=0;
                for (let i=0; i<stat.length; i+=2) {
                    stepCount += stat[i];
                    opCount += stat[i+1];
                }
                const dt = (t-prevStatTime) * 1000;
                MstepSec = stepCount / dt;
                MopSec = opCount / dt;
            });
            data.stat.fill(0);
            data.stat.push();

            const dt = (t-statTime) * 1000;
            const MreactSec = reactCount / dt;
            $('#stat').innerText = 
                `Mstep/sec: ${MstepSec.toFixed(2)}, `+
                `Mop/sec: ${MopSec.toFixed(2)}, ` +
                `Mreact/sec: ${MreactSec.toFixed(2)}, ` +
                `Epoch: ${epochCount}`;
            statTime = t;
            stepCount = 0; opCount = 0; reactCount = 0;
        }        
        await nextFrame();
    }    
}
main();

function step() {
    data.params[0] += 1; // seed
    data.params[1] *= 2; // shuffleStride
    if (data.params[1] >= data.shuffleIdx.length) {
        data.params[1] = 1;
    }
    data.params.push();

    // shuffle
    wgpu.compute(tapeN*2, 128, data, ()=>{
        function main(grid/*:Grid*/) {
            let i = grid.id.x;
            let j = i ^ params.shuffleStride;
            if (i>j) {return;}
            if (hash(params.seed+hash(i))%2 == 0) {return;}
            let t = shuffleIdx[i];
            shuffleIdx[i] = shuffleIdx[j];
            shuffleIdx[j] = t;
        }})

    // noise
    wgpu.compute(tapeN*2, 128, data, ()=>{
        function main(grid/*:Grid*/) {
            var rnd = hash(params.seed+hash(grid.id.x));
            if (rnd%64 != 0) {return;}
            rnd >>= 8;
            let i = grid.id.x * ChunkSize + rnd % ChunkSize;
            let i4 = i/4;
            rnd >>= 8;
            tapes[i4] = insertBits(tapes[i4], rnd&0xff, i%4*8, 8);
        }});

    // bf
    wgpu.compute(tapeN, 64, data, ()=>{
        var/*<private>*/ base0/*:u32*/;
        var/*<private>*/ base1/*:u32*/;
        
        function getByte(p/*:u32*/) /*->u32*/ {
            var i = p%TapeSize;
            i += select(base1-ChunkSize, base0, i<ChunkSize);
            return extractBits(tapes[i/4], i%4*8, 8);
        }
        function setByte(p/*:u32*/, v/*:u32*/) {
            var i = p%TapeSize;
            i += select(base1-ChunkSize, base0, i<ChunkSize);
            let i4 = i/4;
            tapes[i4] = insertBits(tapes[i4], v, i%4*8, 8);
        }
        
        function main(grid/*:Grid*/) {
            let id0 = shuffleIdx[grid.id.x*2];
            let id1 = shuffleIdx[grid.id.x*2+1];
            base0 = id0*ChunkSize;
            base1 = id1*ChunkSize;
            var cnt = Counters();
            let max_step_n = u32(8*1024);
            var i=2; var head0=getByte(0); var head1=getByte(1); var scan=0;
            // var i=0; var head0=u32(0); var head1=u32(0); var scan=0;
            for (; i>=0 && i<TapeSize && cnt.step_n<max_step_n; i+=i32(scan>=0)*2-1) {
                cnt.step_n++;
                let cmd = getByte(u32(i));
                if (scan != 0) {
                    if (cmd == '[') {scan++;}
                    if (cmd == ']') {scan--;}
                } else {
                    cnt.op_n++;
                    let d = getByte(head0);
                    switch (cmd) {
                        case '<': {head0--;} case '>': {head0++;}
                        case '{': {head1--;} case '}': {head1++;}
                        case '+': {setByte(head0, d+1);}
                        case '-': {setByte(head0, d-1);}
                        case '.': {setByte(head1, d);}
                        case ',': {setByte(head0, getByte(head1));}
                        case '[': {if (d==0) {scan=1;}}
                        case ']': {if (d!=0) {scan=-1;}}
                        default:{cnt.op_n--;}
                    }
                }
            }
            stat[grid.id.x].step_n += cnt.step_n;
            stat[grid.id.x].op_n += cnt.op_n;
        }})
        
}

function render() {
    const tex = context.getCurrentTexture();
    const screen = screenGroup.make({screen: tex.createView()});
    const {width, height} = canvas;

    wgpu.compute([width, height], [8,16], data, screen, ()=>{
        const FontSize=vec2u(8, 16);

        function isOP(c/*:u32*/)/*->bool*/ {
            return c=='<'||c=='>'||c=='{'||c=='}'||c=='+'||c=='-'||c=='.'||c==','||c=='['||c==']'||c==0;
        }
        function getByte(xy/*:vec2u*/)/*->u32*/ {
            let i = xy.y*1024 + xy.x;
            return extractBits(tapes[i/4], (i%4)*8, 8);
        }
        function getFont(xy/*:vec2u*/, c/*:u32*/)/*->f32*/ {
            let rc = xy%FontSize;
            return f32(extractBits(font[c*4+rc.y/4], (rc.y%4)*8+7-rc.x, 1));
        }
        function renderText(p/*:vec2u*/) /*->vec4f*/ {
            let i = p/FontSize;
            let i0 = vec2u(i.x/ChunkSize*ChunkSize, i.y);
            let c = getByte(i);
            let head0 = getByte(i0)%TapeSize;
            let head1 = getByte(vec2u(i0.x+1, i0.y))%TapeSize;
            let k = i.x%ChunkSize + select(u32(0), ChunkSize,  p.y%FontSize.y < FontSize.y/2);
            var bg = vec4f(f32(k==head0), 0.0, f32(k==head1), 1.0);
            var bit = getFont(p, select('0', c, c!=0));
            var fg = vec4f(0.0,0.25 + 0.75*f32(isOP(c)),0.0,1.0);
            return vec4f(bg + bit*(fg-bg));
        }
        function main(grid/*:Grid*/) {
            let id = grid.id.xy;
            var color = vec4f(0);
            if (id.x>=1024) {
                color = renderText(id-vec2u(1024, 0));
            } else {
                let i = u32(id.x%ChunkSize);
                color = vec4f(f32(i==0)*0.5);
                if (isOP(getByte(id))) {
                    color = vec4f(0,1,0,1);
                }
            }
            textureStore(screen, id, color);
        }});
}