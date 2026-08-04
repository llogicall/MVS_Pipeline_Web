class PlyViewer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.points = null;
        this.cameraGroup = null;   // 相机视锥体组
        this.animationId = null;
        this.initialized = false;

        this.init();
    }

    async init() {
        try {
            const { Scene, PerspectiveCamera, WebGLRenderer, Color, AmbientLight, DirectionalLight, Vector3 } = await this.loadThree();
            const { OrbitControls } = await this.loadOrbitControls();
            const { PLYLoader } = await this.loadPLYLoader();

            this.scene = new Scene();
            this.scene.background = new Color(0x0a0e14);

            const ambientLight = new AmbientLight(0x404060);
            this.scene.add(ambientLight);

            const dirLight = new DirectionalLight(0xffffff, 1);
            dirLight.position.set(1, 2, 1);
            this.scene.add(dirLight);

            const dirLight2 = new DirectionalLight(0x8888ff, 0.5);
            dirLight2.position.set(-1, -1, -1);
            this.scene.add(dirLight2);

            const rect = this.container.getBoundingClientRect();
            const width = rect.width || 600;
            const height = rect.height || 400;

            this.camera = new PerspectiveCamera(45, width / height, 0.1, 1000);
            this.camera.position.set(3, 2, 3);

            this.renderer = new WebGLRenderer({ antialias: true });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.container.appendChild(this.renderer.domElement);

            this.controls = new OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.1;
            this.controls.target.set(0, 0, 0);

            this.PLYLoader = PLYLoader;
            this.initialized = true;

            this.render();

            window.addEventListener('resize', () => this.onResize());
        } catch (err) {
            console.error('Failed to init Three.js viewer:', err);
            this.container.innerHTML = `<div class="three-placeholder">Three.js 加载失败: ${err.message}</div>`;
        }
    }

    async loadThree() {
        if (window.THREE) return window.THREE;
        const module = await import('three');
        window.THREE = module;
        return module;
    }

    async loadOrbitControls() {
        const module = await import('three/addons/controls/OrbitControls.js');
        return module;
    }

    async loadPLYLoader() {
        const module = await import('three/addons/loaders/PLYLoader.js');
        return module;
    }

    async load(url, token) {
        const placeholder = this.container.querySelector('.three-placeholder');
        if (placeholder) placeholder.remove();

        if (!this.initialized) {
            await new Promise(resolve => {
                const check = () => {
                    if (this.initialized) resolve();
                    else setTimeout(check, 100);
                };
                check();
            });
        }

        // 在 URL 中附加 token 用于认证
        const loadUrl = token
            ? url + (url.includes('?') ? '&' : '?') + `token=${token}`
            : url;

        return new Promise((resolve, reject) => {
            const loader = new this.PLYLoader();
            loader.load(
                loadUrl,
                (geometry) => {
                    this.setGeometry(geometry);
                    resolve();
                },
                undefined,
                (err) => reject(err)
            );
        });
    }

    async loadCameras(camerasUrl, token) {
        if (!this.initialized) {
            await new Promise(resolve => {
                const check = () => {
                    if (this.initialized) resolve();
                    else setTimeout(check, 100);
                };
                check();
            });
        }

        try {
            // 在 URL 中附加 token 用于认证
            const url = token
                ? camerasUrl + (camerasUrl.includes('?') ? '&' : '?') + `token=${token}`
                : camerasUrl;
            const res = await fetch(url);
            if (!res.ok) {
                let msg = `HTTP ${res.status}`;
                try {
                    const err = await res.json();
                    msg = err.detail || msg;
                } catch {}
                throw new Error(msg);
            }
            const data = await res.json();
            if (!data.cameras || data.cameras.length === 0) {
                throw new Error('未找到相机数据');
            }
            this.addCameras(data.cameras);
        } catch (err) {
            console.warn('Failed to load camera poses:', err);
            throw err;  // 向上抛出，让调用方处理
        }
    }

    addCameras(cameras) {
        const THREE = window.THREE;
        const { BufferGeometry, Float32BufferAttribute, Mesh, MeshBasicMaterial, MeshPhongMaterial, Group, Color } = THREE;

        // 移除已有的相机组
        if (this.cameraGroup) {
            this.scene.remove(this.cameraGroup);
            this.cameraGroup = null;
        }

        // 计算场景范围以确定相机图标大小
        const positions = cameras.map(c => c.position);
        const center = this._computeCenter(positions);
        const radius = this._computeRadius(positions, center);
        const frustumScale = Math.max(radius * 0.06, 0.01);

        const group = new Group();

        for (let i = 0; i < cameras.length; i++) {
            const cam = cameras[i];
            const R = cam.rotation; // 3x3 行主序
            const pos = cam.position;

            // 创建金字塔几何体（底面四边形 + 顶点）
            const s = frustumScale;
            const d = frustumScale * 1.8;
            const verts = new Float32Array([
                0, 0, 0,          // 0: apex
                -s, -s, d,        // 1: base BL
                s, -s, d,         // 2: base BR
                s, s, d,          // 3: base TR
                -s, s, d,         // 4: base TL
            ]);
            const idx = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1];

            const geo = new BufferGeometry();
            geo.setAttribute('position', new Float32BufferAttribute(verts, 3));
            geo.setIndex(idx);
            geo.computeVertexNormals();

            // 渐变色：从蓝到红渐变
            const hue = (i / Math.max(cameras.length - 1, 1)) * 0.7; // 0→蓝, 0.7→红
            const color = new Color().setHSL(hue, 0.9, 0.5);

            const mat = new MeshPhongMaterial({
                color: color,
                shininess: 30,
                emissive: color,
                emissiveIntensity: 0.15,
                transparent: true,
                opacity: 0.85,
            });

            const mesh = new Mesh(geo, mat);

            // 构建变换矩阵：local → world
            // 金字塔 local 指向 +Z，需要映射到世界空间中的相机朝向
            // 相机 right = R[0], up = -R[1], forward = R[2]
            const m = new THREE.Matrix4();
            m.set(
                R[0], -R[3], R[6], pos[0],
                R[1], -R[4], R[7], pos[1],
                R[2], -R[5], R[8], pos[2],
                0, 0, 0, 1
            );
            mesh.matrixAutoUpdate = false;
            mesh.matrix.copy(m);

            group.add(mesh);
        }

        this.cameraGroup = group;
        this.scene.add(group);

        // 调整视图相机位置，使相机位姿可见
        this._fitViewToPositions(positions, center, radius);
    }

    _fitViewToPositions(positions, center, radius) {
        if (!this.camera || !this.controls || radius <= 0) return;
        const dist = radius * 2.5;
        this.camera.position.set(
            center[0] + dist,
            center[1] + dist * 0.7,
            center[2] + dist
        );
        this.controls.target.set(center[0], center[1], center[2]);
        this.controls.update();
    }

    toggleCameras(visible) {
        if (this.cameraGroup) {
            this.cameraGroup.visible = visible;
        }
    }

    removeCameras() {
        if (this.cameraGroup) {
            this.scene.remove(this.cameraGroup);
            this.cameraGroup = null;
        }
    }

    _computeCenter(positions) {
        if (positions.length === 0) return [0, 0, 0];
        const sum = [0, 0, 0];
        for (const p of positions) {
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
        }
        return [sum[0] / positions.length, sum[1] / positions.length, sum[2] / positions.length];
    }

    _computeRadius(positions, center) {
        if (positions.length === 0) return 1;
        let maxDistSq = 0;
        for (const p of positions) {
            const dx = p[0] - center[0];
            const dy = p[1] - center[1];
            const dz = p[2] - center[2];
            const dSq = dx * dx + dy * dy + dz * dz;
            if (dSq > maxDistSq) maxDistSq = dSq;
        }
        return Math.sqrt(maxDistSq);
    }

    setGeometry(geometry) {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }

        geometry.computeVertexNormals();

        const { BufferGeometry, Float32BufferAttribute, PointsMaterial, Points } = window.THREE;

        const material = new PointsMaterial({
            size: 0.02,
            vertexColors: geometry.hasAttribute('color'),
            sizeAttenuation: true,
        });

        if (!geometry.hasAttribute('color') && geometry.hasAttribute('position')) {
            const positions = geometry.getAttribute('position');
            const colors = new Float32Array(positions.count * 3);
            for (let i = 0; i < colors.length; i += 3) {
                colors[i] = 0.8;
                colors[i + 1] = 0.8;
                colors[i + 2] = 0.8;
            }
            geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
            material.vertexColors = true;
        }

        this.points = new Points(geometry, material);

        geometry.computeBoundingSphere();
        if (geometry.boundingSphere) {
            const radius = geometry.boundingSphere.radius;
            this.camera.position.set(radius * 2, radius * 1.5, radius * 2);
            this.controls.target.copy(geometry.boundingSphere.center);
            this.controls.update();
        }

        this.scene.add(this.points);
    }

    render() {
        if (!this.initialized) return;
        this.animationId = requestAnimationFrame(() => this.render());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        if (!this.initialized) return;
        const rect = this.container.getBoundingClientRect();
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(rect.width, rect.height);
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.renderer) {
            this.renderer.dispose();
            if (this.renderer.domElement && this.renderer.domElement.parentNode) {
                this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
            }
        }
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }
        this.removeCameras();
    }
}
