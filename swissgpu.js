const removeSpaces = s=>(s?.replaceAll(/\s/gs, ''));

class BindGroup {
    constructor(wgpu, decl) {
        this.wgpu = wgpu;
        this.name2index = {};
        this.layoutDesc = {entries:[]};
        this.decl = decl.replaceAll(/var\s*(<[\w\s,]+>)?\s*(\w+)\s*:([^;]+);/gs, (full, attr, name, type)=>{
            attr = removeSpaces(attr||'');
            if (attr == '<storage>') {
                attr = '<storage,read_write>';
            }
            let item;
            if (attr == '<storage,read_write>') {
                item = {buffer: {type: 'storage'}};
            } else if (attr == '<storage,read>') {
                item = {buffer: {type: 'read-only-storage'}};
            } else if (attr == '<uniform>') {
                item = {buffer: {type: 'uniform'}};
            } else if (type.includes('texture_storage')) {
                let [_, format, access] = removeSpaces(type).match(/<(\w+),(\w+)>/);
                access = access=='write' ? "write-only" : access;
                item = {storageTexture: {access, format}};
            } else {
                return full;
            }
            const index = this.layoutDesc.entries.length;
            this.name2index[name] = index;
            this.layoutDesc.entries.push({binding:index, visibility: GPUShaderStage.COMPUTE, ...item});      
            return `@group($group) @binding(${index}) var${attr} ${name}: ${type};`
        });
        this.layout = wgpu.device.createBindGroupLayout(this.layoutDesc);
    }

    make(data) {
        const desc = {layout: this.layout, entries:[]};
        const fields = {}
        for (const name in data) {
            const src = data[name];
            const index = this.name2index[name];
            let item = {binding: index};
            if (this.layoutDesc.entries[index].buffer) {
                const [buffer, cpuBuffer] = this.wgpu.createBuffer(name, src);
                item.resource = {buffer};
                fields[name] = cpuBuffer;
            } else {
                item.resource = src;
            }
            desc.entries.push(item);
        }
        const group = this.wgpu.device.createBindGroup(desc);
        group._layout = this;
        group._desc = desc;
        for (const name in fields) {
            group[name] = fields[name];
        }
        return group;
    }
}

class WGPU {
    constructor(device, include='') {
        this.device = device;
        this.include = include;
        this.pipelines = {};
    }

    group(decl) {return new BindGroup(this, decl);}

    compute(total, block, ...args) {
        let   [gx,gy=1,gz=1] = (typeof total == 'number') ? [total] : total;
        const [bx,by=1,bz=1] = (typeof block == 'number') ? [block] : block;
        const grid = [Math.ceil(gx/bx), Math.ceil(gy/by), Math.ceil(gz/bz)];

        const {device} = this;
        let code = args.pop();
        if (code instanceof Function) { // this is a super speculative JS->WGSL conversion
            code = code.toString()
                .replaceAll('function', 'fn')
                .replaceAll(/\/\*([&@:<-][^*]*)\*\//g, (_,s)=>s)
                .replace(/^\(\)=>{/, '')
                .replace(/}$/, '');
        }
        code = code.replace(/\bfn\s+main\b/, `@compute @workgroup_size(${bx},${by},${bz}) fn main`);
        const chunks = [this.include,
            ...args.map((d,i)=>d._layout.decl.replaceAll('$group', i)), code];
        const joined = chunks.join('\n');
        if (!(joined in this.pipelines)) {
            const processedCode = joined.replace(/'.'/g, c=>c.charCodeAt(1));
            const module = device.createShaderModule({code: processedCode});
            const layout = device.createPipelineLayout({bindGroupLayouts:args.map(d=>d._layout.layout)});
            const pipeline = device.createComputePipeline({layout, compute: { module, entryPoint:'main'}});
            this.pipelines[joined] = pipeline;
        }
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines[joined]);
        args.forEach((group, i)=>pass.setBindGroup(i, group));
        pass.dispatchWorkgroups(...grid);
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    createBuffer(label, cpuBuffer) {
        const size = cpuBuffer.byteLength;
        const device = this.device;
        const buffer = device.createBuffer({label, size, usage:GPUBufferUsage.UNIFORM |
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
        cpuBuffer._gpuBuffer = buffer;
        cpuBuffer.push = ()=>device.queue.writeBuffer(buffer, 0, cpuBuffer);
        cpuBuffer.pull = async()=>this.pullBuffer(cpuBuffer);
        cpuBuffer.push();
        return [buffer, cpuBuffer];
    }

    async pullBuffer(cpuBuffer) {
        const size = cpuBuffer.byteLength;
        const device = this.device;
        if (!cpuBuffer._shadow) {
            cpuBuffer._shadow = device.createBuffer({size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
        }
        const shadow = cpuBuffer._shadow;
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(cpuBuffer._gpuBuffer, 0, shadow, 0, size);
        device.queue.submit([encoder.finish()]);
        await shadow.mapAsync(GPUMapMode.READ);
        const src = new cpuBuffer.constructor(shadow.getMappedRange());
        cpuBuffer.set(src);
        shadow.unmap();
        return cpuBuffer;
    }
}
