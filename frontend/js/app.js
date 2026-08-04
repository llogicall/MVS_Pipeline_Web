class MvsApi {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.token = localStorage.getItem('auth_token') || null;
    }

    _headers(extra = {}) {
        const h = { ...extra };
        if (this.token) {
            h['Authorization'] = `Bearer ${this.token}`;
        }
        return h;
    }

    async _fetch(url, options = {}) {
        const opts = { ...options };
        opts.headers = this._headers(opts.headers);
        const res = await fetch(url, opts);
        if (res.status === 401) {
            // Token expired or invalid
            this.token = null;
            localStorage.removeItem('auth_token');
            if (window.app) window.app.onAuthExpired();
        }
        return res;
    }

    async login(username, password) {
        const res = await fetch(`${this.baseUrl}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (res.ok && data.token) {
            this.token = data.token;
            localStorage.setItem('auth_token', data.token);
        }
        return { ok: res.ok, ...data };
    }

    async register(username, password) {
        const res = await fetch(`${this.baseUrl}/api/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (res.ok && data.token) {
            this.token = data.token;
            localStorage.setItem('auth_token', data.token);
        }
        return { ok: res.ok, ...data };
    }

    logout() {
        this.token = null;
        localStorage.removeItem('auth_token');
    }

    async getMe() {
        const res = await this._fetch(`${this.baseUrl}/api/auth/me`);
        if (!res.ok) return null;
        return res.json();
    }

    async runPipeline(params) {
        const res = await this._fetch(`${this.baseUrl}/api/pipeline/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try {
                const err = await res.json();
                msg = err.detail || msg;
            } catch {
                // 如果不是 JSON，尝试获取文本
                try {
                    const text = await res.text();
                    msg = text.substring(0, 200) || msg;
                } catch {}
            }
            throw new Error(msg);
        }
        return res.json();
    }

    async getStatus(taskId) {
        const res = await this._fetch(`${this.baseUrl}/api/pipeline/status/${taskId}`);
        return res.json();
    }

    async getLogs(taskId, offset = 0) {
        const res = await this._fetch(`${this.baseUrl}/api/pipeline/logs/${taskId}?offset=${offset}`);
        return res.json();
    }

    async cancelTask(taskId) {
        const res = await this._fetch(`${this.baseUrl}/api/pipeline/cancel/${taskId}`, { method: 'POST' });
        return res.json();
    }

    async getPlyInfo(workspaceDir) {
        const params = workspaceDir ? `?workspace_dir=${encodeURIComponent(workspaceDir)}` : '';
        const res = await this._fetch(`${this.baseUrl}/api/files/ply/info${params}`);
        if (!res.ok) return { found: false };
        return res.json();
    }

    getPlyUrl(workspaceDir) {
        const params = workspaceDir ? `?workspace_dir=${encodeURIComponent(workspaceDir)}` : '';
        return `${this.baseUrl}/api/files/ply${params}`;
    }

    getCamerasUrl(workspaceDir, sceneName) {
        const params = new URLSearchParams();
        if (workspaceDir) params.set('workspace_dir', workspaceDir);
        if (sceneName) params.set('scene_name', sceneName);
        const qs = params.toString();
        return `${this.baseUrl}/api/files/cameras${qs ? '?' + qs : ''}`;
    }

    getSparsePlyUrl(workspaceDir, sceneName) {
        const params = new URLSearchParams();
        if (workspaceDir) params.set('workspace_dir', workspaceDir);
        if (sceneName) params.set('scene_name', sceneName);
        const qs = params.toString();
        return `${this.baseUrl}/api/files/sparse_ply${qs ? '?' + qs : ''}`;
    }

    async getPipelineStatus(workspaceDir, sceneName) {
        const params = new URLSearchParams();
        if (workspaceDir) params.set('workspace_dir', workspaceDir);
        if (sceneName) params.set('scene_name', sceneName);
        const qs = params.toString();
        const res = await this._fetch(`${this.baseUrl}/api/files/pipeline/status${qs ? '?' + qs : ''}`);
        if (!res.ok) return {};
        return res.json();
    }

    async getUploadedImages(uploadDir) {
        const params = uploadDir ? `?upload_dir=${encodeURIComponent(uploadDir)}` : '';
        const res = await this._fetch(`${this.baseUrl}/api/files/upload/images/list${params}`);
        if (!res.ok) return { images: [], count: 0 };
        return res.json();
    }

    async getMvsMethods() {
        const res = await this._fetch(`${this.baseUrl}/api/files/mvs/methods`);
        if (!res.ok) return { methods: [] };
        return res.json();
    }

    async getMvsCheckpoints(method) {
        const params = method ? `?method=${encodeURIComponent(method)}` : '';
        const res = await this._fetch(`${this.baseUrl}/api/files/mvs/checkpoints${params}`);
        if (!res.ok) return { checkpoints: [] };
        return res.json();
    }

    async getDepthResults(workspaceDir) {
        const params = workspaceDir ? `?workspace_dir=${encodeURIComponent(workspaceDir)}` : '';
        const res = await this._fetch(`${this.baseUrl}/api/files/depth_results${params}`);
        if (!res.ok) return { results: [] };
        return res.json();
    }

    async getDepthResultPointclouds(workspaceDir, depthResultId) {
        const params = new URLSearchParams();
        if (workspaceDir) params.set('workspace_dir', workspaceDir);
        const url = `${this.baseUrl}/api/files/depth_results/${encodeURIComponent(depthResultId)}/pointclouds?${params}`;
        const res = await this._fetch(url);
        if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try {
                const err = await res.json();
                msg = err.detail || msg;
            } catch {}
            throw new Error(msg);
        }
        return res.json();
    }

    getDepthResultPointcloudUrl(workspaceDir, depthResultId, pointcloudPath) {
        const params = new URLSearchParams();
        if (workspaceDir) params.set('workspace_dir', workspaceDir);
        params.set('path', pointcloudPath);
        return `${this.baseUrl}/api/files/depth_results/${encodeURIComponent(depthResultId)}/pointcloud?${params}`;
    }

    async runDeepFusion(params) {
        const res = await this._fetch(`${this.baseUrl}/api/pipeline/deep-fusion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try {
                const err = await res.json();
                msg = err.detail || msg;
            } catch {}
            throw new Error(msg);
        }
        return res.json();
    }

    async runPointCloudProcess(params) {
        const res = await this._fetch(`${this.baseUrl}/api/pipeline/pointcloud/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try {
                const err = await res.json();
                msg = err.detail || msg;
            } catch {}
            throw new Error(msg);
        }
        return res.json();
    }

    connectWs(taskId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}${this.baseUrl}/api/pipeline/ws/logs/${taskId}`;
        return new WebSocket(wsUrl);
    }

    // ---- 项目管理 ----
    async createProject(name) {
        const res = await this._fetch(`${this.baseUrl}/api/projects/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        return res.json();
    }

    async getProjects() {
        const res = await this._fetch(`${this.baseUrl}/api/projects`);
        if (!res.ok) return { projects: [] };
        return res.json();
    }

    async getProject(id) {
        const res = await this._fetch(`${this.baseUrl}/api/projects/${id}`);
        if (!res.ok) return null;
        return res.json();
    }

    async extractVideoFrames(params) {
        const res = await this._fetch(`${this.baseUrl}/api/files/upload/video/extract_frames`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try {
                const err = await res.json();
                msg = err.detail || msg;
            } catch {}
            throw new Error(msg);
        }
        return res.json();
    }

    async updateProject(id, data) {
        const res = await this._fetch(`${this.baseUrl}/api/projects/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return res.json();
    }

    async deleteProject(id, deleteFiles = false) {
        const url = `${this.baseUrl}/api/projects/${id}${deleteFiles ? '?delete_files=true' : ''}`;
        const res = await this._fetch(url, {
            method: 'DELETE',
        });
        return res.json();
    }
}

class MvsApp {
    constructor() {
        this.api = new MvsApi();
        this.currentTaskId = null;
        this.ws = null;
        this.logLines = [];
        this.camerasVisible = false;
        this.isLoggedIn = false;
        this.viewer = null;
        this._currentProgressStage = 0;
        this.currentTaskDepthResultId = null;
        this._preferredDepthResultId = '';

        this.elements = {
            logs: document.getElementById('log_output'),
            statusBadge: document.getElementById('task_status_badge'),
            plyInfo: document.getElementById('ply_info'),
            downloadBtn: document.getElementById('download_ply_btn'),
            toggleCamerasBtn: document.getElementById('toggle_cameras_btn'),
            showCamerasBtn: document.getElementById('show_cameras_btn'),
            showSparseBtn: document.getElementById('show_sparse_btn'),
            cancelBtn: document.getElementById('cancel_task_btn'),
            clearLogsBtn: document.getElementById('clear_logs_btn'),
            threeContainer: document.getElementById('three_container'),
            browseImagesBtn: document.getElementById('browse_images_btn'),
            // 视频上传
            videoUpload: document.getElementById('video_upload'),
            videoSettings: document.getElementById('video_settings'),
            videoFps: document.getElementById('video_fps'),
            videoMaxFrames: document.getElementById('video_max_frames'),
            extractFramesBtn: document.getElementById('extract_frames_btn'),
            // 相机采集
            cameraOpenBtn: document.getElementById('camera_open_btn'),
            cameraStartBtn: document.getElementById('camera_start_btn'),
            cameraStopBtn: document.getElementById('camera_stop_btn'),
            cameraPreview: document.getElementById('camera_preview'),
            cameraPreviewContainer: document.getElementById('camera_preview_container'),
            cameraInterval: document.getElementById('camera_interval'),
            cameraStatus: document.getElementById('camera_status'),
            // Auth
            authArea: document.getElementById('auth_area'),
            authUsername: document.getElementById('auth_username_display'),
            loginOverlay: document.getElementById('login_overlay'),
            loginUsername: document.getElementById('login_username'),
            loginPassword: document.getElementById('login_password'),
            loginBtn: document.getElementById('login_btn'),
            registerBtn: document.getElementById('register_btn'),
            logoutBtn: document.getElementById('logout_btn'),
            loginError: document.getElementById('login_error'),
            loginHint: document.getElementById('login_hint'),
            appContainer: document.getElementById('app_container'),
            // Project
            projectArea: document.getElementById('project_area'),
            projectSelector: document.getElementById('project_selector'),
            newProjectBtn: document.getElementById('new_project_btn'),
            deleteProjectBtn: document.getElementById('delete_project_btn'),
            // MVS
            mvsRoot: document.getElementById('mvs_root'),
            checkpoint: document.getElementById('checkpoint'),
            // 点云处理
            deepFusionConf: document.getElementById('conf'),
            deepFusionThres: document.getElementById('thres_view'),
            fusionNumWorker: document.getElementById('num_worker'),
            btnDeepFusion: document.getElementById('btn_deep_fusion'),
            voxelSize: document.getElementById('voxel_size'),
            btnVoxel: document.getElementById('btn_voxel_downsample'),
            statNbNeighbors: document.getElementById('stat_nb_neighbors'),
            statStdRatio: document.getElementById('stat_std_ratio'),
            btnStatFilter: document.getElementById('btn_stat_filter'),
            radiusNbPoints: document.getElementById('radius_nb_points'),
            radiusValue: document.getElementById('radius_value'),
            btnRadiusFilter: document.getElementById('btn_radius_filter'),
            // 深度估计来源
            depthSource: document.getElementById('depth_source'),
            pointcloudInput: document.getElementById('pointcloud_input'),
            pointcloudTasks: document.getElementById('pointcloud_tasks'),
            refreshDepthBtn: document.getElementById('refresh_depth_btn'),
        };

        this.currentProjectId = null;  // 当前项目 ID
        this._logLines = [];            // 内存中的日志行
        this._saveTimer = null;         // 自动保存防抖定时器

        this.bindEvents();
        this.initViewer();
    }

    getConfig() {
        // 共享字段：从当前激活的标签取值
        const activeTab = document.querySelector('.tab-btn.active')?.dataset?.tab || 'colmap';
        const wsId = activeTab === 'colmap' ? 'colmap_workspace_dir' : 'mvs_workspace_dir';
        return {
            workspace_dir: document.getElementById(wsId).value || 'runs',
            scene_name: 'scene',  // 固定值，由项目系统管理
            mvs_root: document.getElementById('mvs_root').value || '',
            image_dir: document.getElementById('image_dir').value || '',
            checkpoint: document.getElementById('checkpoint').value || '',
            interval_scale: parseFloat(document.getElementById('interval_scale').value) || 1.06,
            num_view: parseInt(document.getElementById('num_view').value) || 5,
            max_h: parseInt(document.getElementById('max_h').value) || 864,
            max_w: parseInt(document.getElementById('max_w').value) || 1152,
            conf: parseFloat(document.getElementById('conf').value) || 0.9,
            thres_view: parseInt(document.getElementById('thres_view').value) || 5,
            num_worker: parseInt(document.getElementById('num_worker').value) || 4,
            filter_method: document.getElementById('filter_method').value,
            camera_model: document.getElementById('camera_model').value,
            single_camera: document.getElementById('single_camera').checked,
            use_gpu: document.getElementById('use_gpu').checked,
            sfm_max_image_size: parseInt(document.getElementById('sfm_max_image_size').value) || 800,
            sfm_max_num_features: parseInt(document.getElementById('sfm_max_num_features').value) || 4096,
            depth_result_id: this.elements.depthSource?.value || '',
        };
    }

    bindEvents() {
        document.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', () => {
                this.startPipeline(btn.dataset.action);
            });
        });

        this.elements.cancelBtn.addEventListener('click', () => this.cancelTask());
        this.elements.clearLogsBtn.addEventListener('click', () => this.clearLogs());
        this.elements.downloadBtn.addEventListener('click', () => this.downloadPly());
        this.elements.toggleCamerasBtn.addEventListener('click', () => this.toggleCameras());
        this.elements.showCamerasBtn.addEventListener('click', () => this.showCameras());
        this.elements.showSparseBtn.addEventListener('click', () => this.showSparsePly());

        // 标签切换
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        // 主题切换
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', () => this.setTheme(btn.dataset.theme));
        });

        // 同步共享字段（工作目录 + 场景名称）
        this._syncFields = { colmap: {}, mvs: {} };
        ['colmap_workspace_dir', 'mvs_workspace_dir'].forEach(id => {
            document.getElementById(id).addEventListener('input', () => {
                this.syncSharedField('workspace_dir');
                this.scheduleSaveProject();
            });
        });

        // 参数变更自动保存（防抖）
        document.querySelectorAll('.tab-content input, .tab-content select').forEach(el => {
            el.addEventListener('change', () => this.scheduleSaveProject());
        });

        // 项目事件
        this.elements.projectSelector.addEventListener('change', () => {
            const id = this.elements.projectSelector.value;
            if (id) this.selectProject(id);
        });
        this.elements.newProjectBtn.addEventListener('click', () => this.createProject());
        this.elements.deleteProjectBtn.addEventListener('click', () => this.deleteProject());

        // MVS 方法目录变更时更新 checkpoint 下拉框
        this.elements.mvsRoot.addEventListener('change', () => this.updateCheckpointList());

        // 深度融合按钮
        if (this.elements.btnDeepFusion) {
            this.elements.btnDeepFusion.addEventListener('click', () => this.runDeepFusion());
        }
        // Open3D 点云后处理
        if (this.elements.btnVoxel) {
            this.elements.btnVoxel.addEventListener('click', () => this.runPointCloudProcess('voxel'));
        }
        if (this.elements.btnStatFilter) {
            this.elements.btnStatFilter.addEventListener('click', () => this.runPointCloudProcess('statistical'));
        }
        if (this.elements.btnRadiusFilter) {
            this.elements.btnRadiusFilter.addEventListener('click', () => this.runPointCloudProcess('radius'));
        }

        this.elements.browseImagesBtn.addEventListener('click', () => this.browseImages());

        document.getElementById('image_upload').addEventListener('change', async (e) => {
            await this.uploadImages(e.target.files);
        });

        // 视频上传事件
        if (this.elements.videoUpload) {
            this.elements.videoUpload.addEventListener('change', async (e) => {
                await this.handleVideoUpload(e.target.files);
            });
        }
        if (this.elements.extractFramesBtn) {
            this.elements.extractFramesBtn.addEventListener('click', () => this.extractVideoFrames());
        }

        // 相机采集事件
        if (this.elements.cameraOpenBtn) {
            this.elements.cameraOpenBtn.addEventListener('click', () => this.openCamera());
        }
        if (this.elements.cameraStartBtn) {
            this.elements.cameraStartBtn.addEventListener('click', () => this.startCameraCapture());
        }
        if (this.elements.cameraStopBtn) {
            this.elements.cameraStopBtn.addEventListener('click', () => this.stopCameraCapture());
        }
        // 深度估计来源刷新
        if (this.elements.refreshDepthBtn) {
            this.elements.refreshDepthBtn.addEventListener('click', () => this.loadDepthResults());
        }
        if (this.elements.depthSource) {
            this.elements.depthSource.addEventListener('change', () => {
                this._preferredDepthResultId = this.elements.depthSource.value;
                this.loadPointCloudState(this._preferredDepthResultId, true);
                this.scheduleSaveProject();
            });
        }
        if (this.elements.pointcloudInput) {
            this.elements.pointcloudInput.addEventListener('change', () => this.loadSelectedPointCloud());
        }

        // Auth events
        this.elements.loginBtn.addEventListener('click', () => this.login());
        this.elements.registerBtn.addEventListener('click', () => this.register());
        this.elements.logoutBtn.addEventListener('click', () => this.logout());
        this.elements.loginUsername.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.login();
        });
        this.elements.loginPassword.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.login();
        });
    }

    async browseImages() {
        const input = document.getElementById('image_upload');
        input.click();
    }

    async uploadImages(fileList) {
        if (!fileList || fileList.length === 0) return;

        const allFiles = Array.from(fileList).filter(f => {
            const ext = f.name.split('.').pop().toLowerCase();
            return ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'].includes(ext);
        });

        if (allFiles.length === 0) {
            this.appendLog('错误: 未找到有效图像文件');
            return;
        }

        const BATCH_SIZE = 20;  // 每批上传的文件数
        let uploadPath = null;
        let totalUploaded = 0;

        for (let i = 0; i < allFiles.length; i += BATCH_SIZE) {
            const batch = allFiles.slice(i, i + BATCH_SIZE);
            const formData = new FormData();
            batch.forEach(f => formData.append('files', f));

            let url = '/api/files/upload/images';
            if (uploadPath) {
                url += `?upload_dir=${encodeURIComponent(uploadPath)}`;
            }

            try {
                const res = await fetch(url, {
                    method: 'POST',
                    body: formData,
                    headers: this.api.token ? { 'Authorization': `Bearer ${this.api.token}` } : {},
                });
                if (!res.ok) {
                    let msg = `HTTP ${res.status}`;
                    try {
                        const body = await res.text();
                        try {
                            const err = JSON.parse(body);
                            msg = err.detail || err.message || msg;
                        } catch {
                            msg = body || msg;
                        }
                    } catch {
                        // res.text() 也失败了，使用默认消息
                    }
                    throw new Error(msg);
                }
                const data = await res.json();
                uploadPath = data.path;
                totalUploaded += data.count;
                this.appendLog(`上传进度: ${totalUploaded}/${allFiles.length}`);
            } catch (err) {
                this.appendLog(`上传失败: ${err.message}`);
                return;
            }
        }

        if (uploadPath) {
            document.getElementById('image_dir').value = uploadPath;
        }
        this.appendLog(`上传完成: ${totalUploaded} 张图像到 ${uploadPath}`);
        // 保存项目（上传目录）
        if (uploadPath) {
            this.saveProject({ image_upload_dir: uploadPath });
        }
        // 隐藏上传控件，显示完成状态
        const uploadGroup = document.getElementById('upload_group');
        if (uploadGroup) {
            uploadGroup.innerHTML = `<label>图像状态</label><span class="upload-status">已上传 ${totalUploaded} 张</span>`;
        }
        // 显示图像预览画廊
        this.loadImageGallery(uploadPath);
    }

    // ====================================================================
    // 视频上传与帧提取
    // ====================================================================
    async handleVideoUpload(fileList) {
        if (!fileList || fileList.length === 0) return;

        const videoFile = fileList[0];
        this.appendLog(`正在上传视频: ${videoFile.name} (${(videoFile.size / 1024 / 1024).toFixed(1)} MB)`);
        this.setButtonsDisabled(true);
        this.setStatus('running', '上传中');

        const formData = new FormData();
        formData.append('file', videoFile);

        try {
            const res = await fetch('/api/files/upload/video', {
                method: 'POST',
                body: formData,
                headers: this.api.token ? { 'Authorization': `Bearer ${this.api.token}` } : {},
            });

            if (!res.ok) {
                let msg = `HTTP ${res.status}`;
                try {
                    const err = await res.json();
                    msg = err.detail || msg;
                } catch {}
                throw new Error(msg);
            }

            const data = await res.json();
            this._lastVideoPath = data.video_path;
            this._lastUploadDir = data.upload_dir;
            this.appendLog(`视频上传成功: ${data.video_path}`);

            // 显示帧提取设置
            if (this.elements.videoSettings) {
                this.elements.videoSettings.style.display = 'block';
            }

            this.setButtonsDisabled(false);
            this.setStatus('idle', '就绪');

            // 保存项目
            this.saveProject({ video_path: data.video_path });

        } catch (err) {
            this.appendLog(`视频上传失败: ${err.message}`);
            this.setButtonsDisabled(false);
            this.setStatus('failed', '失败');
        }
    }

    async extractVideoFrames() {
        const videoPath = this._lastVideoPath;
        if (!videoPath) {
            this.appendLog('请先上传视频');
            return;
        }

        const fps = parseFloat(this.elements.videoFps?.value) || 1;
        const maxFrames = parseInt(this.elements.videoMaxFrames?.value, 10) || 200;

        this.appendLog(`开始提取帧 (fps=${fps}, max_frames=${maxFrames})...`);
        this.setButtonsDisabled(true);
        this.setStatus('running', '提取中');

        try {
            const result = await this.api.extractVideoFrames({
                video_path: videoPath,
                output_dir: '',
                fps: fps,
                max_frames: maxFrames,
            });

            this.appendLog(`帧提取完成: ${result.frame_count} 帧`);
            this.appendLog(`视频 FPS: ${result.fps}, 时长: ${result.duration}s`);
            this.appendLog(`帧目录: ${result.frames_dir}`);

            // 自动设置图像目录
            document.getElementById('image_dir').value = result.frames_dir;

            // 更新上传状态
            const uploadGroup = document.getElementById('upload_group');
            if (uploadGroup) {
                uploadGroup.innerHTML = `<label>图像状态</label><span class="upload-status">从视频提取 ${result.frame_count} 帧</span>`;
            }

            // 隐藏帧提取设置
            if (this.elements.videoSettings) {
                this.elements.videoSettings.style.display = 'none';
            }

            // 显示图像预览画廊
            this.loadImageGallery(result.frames_dir);

            this.saveProject({ image_dir: result.frames_dir });
            this.setButtonsDisabled(false);
            this.setStatus('idle', '就绪');

        } catch (err) {
            this.appendLog(`帧提取失败: ${err.message}`);
            this.setButtonsDisabled(false);
            this.setStatus('failed', '失败');
        }
    }

    // ====================================================================
    // 相机采集
    // ====================================================================
    async openCamera() {
        try {
            this._cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
                audio: false,
            });
            if (this.elements.cameraPreview) {
                this.elements.cameraPreview.srcObject = this._cameraStream;
            }
            if (this.elements.cameraPreviewContainer) {
                this.elements.cameraPreviewContainer.style.display = 'block';
            }
            this.elements.cameraStartBtn.disabled = false;
            this.elements.cameraOpenBtn.disabled = true;
            this._cameraCapturedFrames = [];
            this._updateCameraStatus('相机已打开，点击「开始采集」');
            this.appendLog('相机已打开');
        } catch (err) {
            this.appendLog(`打开相机失败: ${err.message}`);
        }
    }

    async startCameraCapture() {
        if (!this._cameraStream) {
            this.appendLog('请先打开相机');
            return;
        }

        this._cameraCapturing = true;
        this._cameraCapturedFrames = this._cameraCapturedFrames || [];
        this.elements.cameraStartBtn.disabled = true;
        this.elements.cameraStopBtn.disabled = false;

        const interval = (parseFloat(this.elements.cameraInterval?.value) || 1) * 1000;
        this._updateCameraStatus('采集中...');
        this.appendLog(`开始采集，间隔 ${interval / 1000} 秒`);

        const video = this.elements.cameraPreview;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const captureFrame = async () => {
            if (!this._cameraCapturing) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            canvas.toBlob(async (blob) => {
                if (!blob || !this._cameraCapturing) return;
                this._cameraCapturedFrames.push(blob);
                this._updateCameraStatus(`已采集 ${this._cameraCapturedFrames.length} 帧`);
            }, 'image/jpeg', 0.95);

            if (this._cameraCapturing) {
                this._cameraCaptureTimer = setTimeout(captureFrame, interval);
            }
        };

        captureFrame();
    }

    async stopCameraCapture() {
        this._cameraCapturing = false;
        if (this._cameraCaptureTimer) {
            clearTimeout(this._cameraCaptureTimer);
            this._cameraCaptureTimer = null;
        }
        this.elements.cameraStartBtn.disabled = false;
        this.elements.cameraStopBtn.disabled = true;

        const frames = this._cameraCapturedFrames || [];
        this._updateCameraStatus(`采集完成，共 ${frames.length} 帧`);

        if (frames.length === 0) {
            this.appendLog('未采集到任何帧');
            return;
        }

        this.appendLog(`停止采集，共 ${frames.length} 帧，开始上传...`);

        const BATCH_SIZE = 20;
        let uploadPath = null;
        let totalUploaded = 0;

        for (let i = 0; i < frames.length; i += BATCH_SIZE) {
            const batch = frames.slice(i, i + BATCH_SIZE);
            const formData = new FormData();
            batch.forEach((blob, idx) => {
                const filename = `${String(totalUploaded + idx).padStart(8, '0')}.jpg`;
                formData.append('files', blob, filename);
            });

            let url = '/api/files/upload/images';
            if (uploadPath) {
                url += `?upload_dir=${encodeURIComponent(uploadPath)}`;
            }

            try {
                const res = await fetch(url, {
                    method: 'POST',
                    body: formData,
                    headers: this.api.token ? { 'Authorization': `Bearer ${this.api.token}` } : {},
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                uploadPath = data.path;
                totalUploaded += data.count;
                this._updateCameraStatus(`上传中 ${totalUploaded}/${frames.length}`);
            } catch (err) {
                this.appendLog(`帧上传失败: ${err.message}`);
                return;
            }
        }

        if (uploadPath) {
            document.getElementById('image_dir').value = uploadPath;
        }
        this.appendLog(`相机采集完成: ${totalUploaded} 帧已上传到 ${uploadPath}`);

        const uploadGroup = document.getElementById('upload_group');
        if (uploadGroup) {
            uploadGroup.innerHTML = `<label>图像状态</label><span class="upload-status">相机采集 ${totalUploaded} 帧</span>`;
        }
        this.loadImageGallery(uploadPath);
        if (uploadPath) {
            this.saveProject({ image_upload_dir: uploadPath });
        }

        // 关闭相机
        this._closeCamera();
    }

    _closeCamera() {
        if (this._cameraStream) {
            this._cameraStream.getTracks().forEach(track => track.stop());
            this._cameraStream = null;
        }
        if (this.elements.cameraPreview) {
            this.elements.cameraPreview.srcObject = null;
        }
        this.elements.cameraOpenBtn.disabled = false;
        this.elements.cameraStartBtn.disabled = true;
        this.elements.cameraStopBtn.disabled = true;
    }

    _updateCameraStatus(text) {
        if (this.elements.cameraStatus) {
            this.elements.cameraStatus.textContent = text;
        }
    }

    // ====================================================================
    // 图像预览画廊
    // ====================================================================
    async loadImageGallery(uploadDir) {
        const gallery = document.getElementById('image_gallery');
        const scroll = document.getElementById('gallery_scroll');
        if (!gallery || !scroll) return;

        try {
            const data = await this.api.getUploadedImages(uploadDir);
            if (!data.images || data.images.length === 0) {
                gallery.style.display = 'none';
                return;
            }

            scroll.innerHTML = '';
            const token = this.api.token || '';
            data.images.forEach(img => {
                const thumb = document.createElement('img');
                // 在 URL 中附加 token，供 <img> 标签认证使用
                const fullUrl = img.url + (img.url.includes('?') ? '&' : '?') + `token=${token}`;
                thumb.src = fullUrl;
                thumb.alt = img.name;
                thumb.title = img.name;
                thumb.className = 'sidebar-thumb';
                thumb.loading = 'lazy';
                thumb.dataset.fullUrl = fullUrl;
                // 悬浮预览
                thumb.addEventListener('mouseenter', (e) => this._showThumbPreview(e, fullUrl));
                thumb.addEventListener('mouseleave', () => this._hideThumbPreview());
                scroll.appendChild(thumb);
            });

            gallery.style.display = 'block';
        } catch (err) {
            this.appendLog(`加载图像预览失败: ${err.message}`);
        }
    }

    _showThumbPreview(event, imgUrl) {
        let overlay = document.getElementById('thumb_preview_overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'thumb_preview_overlay';
            overlay.className = 'thumb-preview-overlay';
            document.body.appendChild(overlay);
        }
        // 在 overlay 中创建或复用 img
        let img = overlay.querySelector('img');
        if (!img) {
            img = document.createElement('img');
            overlay.appendChild(img);
        }
        img.src = imgUrl;
        img.onload = () => {
            overlay.classList.add('show');
            // 定位：跟随鼠标，偏移 20px
            const x = event.clientX + 20;
            const y = event.clientY + 20;
            overlay.style.left = Math.min(x, window.innerWidth - 620) + 'px';
            overlay.style.top = Math.min(y, window.innerHeight - 620) + 'px';
        };
    }

    _hideThumbPreview() {
        const overlay = document.getElementById('thumb_preview_overlay');
        if (overlay) overlay.classList.remove('show');
    }

    async startPipeline(action) {
        const config = this.getConfig();
        config.action = action;

        this.setButtonsDisabled(true);
        this.setStatus('running', '运行中');
        this.logLines = [];
        this.elements.logs.textContent = '';
        this.resetProgress();  // 重置进度条
        this.appendLog(`启动管道: ${action}\n`);
        this.appendLog(`配置: ${JSON.stringify(config, null, 2)}\n`);

        try {
            const result = await this.api.runPipeline(config);
            this.currentTaskId = result.task_id;
            this.appendLog(`任务 ID: ${result.task_id}\n`);
            this.connectWebSocket(result.task_id);
        } catch (err) {
            this.appendLog(`启动失败: ${err.message}`);
            this.setStatus('failed', '失败');
            this.setButtonsDisabled(false);
        }
    }

    connectWebSocket(taskId) {
        if (this.ws) {
            this.ws.close();
        }

        this.ws = this.api.connectWs(taskId);

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.logs) {
                data.logs.forEach(log => this.appendLog(log));
            }

            if (data.status) {
                this.setStatus(data.status, data.status);
                if (data.status === 'completed') {
                    this.onTaskComplete(data);
                } else if (data.status === 'failed') {
                    this.onTaskFailed(data);
                } else if (data.status === 'cancelled') {
                    this.appendLog('\n任务已取消');
                    this.setButtonsDisabled(false);
                }
            }
        };

        this.ws.onerror = () => {
            this.appendLog('\nWebSocket 连接错误，切换为轮询模式');
            this.startPolling(taskId);
        };

        this.ws.onclose = () => {
            if (this.currentTaskId === taskId) {
                this.pollTaskFinalStatus(taskId);
            }
        };
    }

    async startPolling(taskId) {
        while (this.currentTaskId === taskId) {
            try {
                const [statusRes, logsRes] = await Promise.all([
                    this.api.getStatus(taskId),
                    this.api.getLogs(taskId, this.logLines.length),
                ]);

                if (logsRes.logs) {
                    logsRes.logs.forEach(log => this.appendLog(log));
                }

                if (statusRes.status === 'completed' || statusRes.status === 'failed' || statusRes.status === 'cancelled') {
                    this.setStatus(statusRes.status, statusRes.status);
                    if (statusRes.status === 'completed') {
                        this.onTaskComplete(statusRes);
                    } else if (statusRes.status === 'failed') {
                        this.onTaskFailed(statusRes);
                    }
                    this.setButtonsDisabled(false);
                    break;
                }
            } catch (err) {
                this.appendLog(`轮询错误: ${err.message}`);
            }
            await new Promise(r => setTimeout(r, 500));
        }
    }

    async pollTaskFinalStatus(taskId) {
        for (let i = 0; i < 30; i++) {
            await new Promise(r => setTimeout(r, 500));
            try {
                const statusRes = await this.api.getStatus(taskId);
                if (['completed', 'failed', 'cancelled'].includes(statusRes.status)) {
                    this.setStatus(statusRes.status, statusRes.status);
                    const logsRes = await this.api.getLogs(taskId, this.logLines.length);
                    if (logsRes.logs) {
                        logsRes.logs.forEach(log => this.appendLog(log));
                    }
                    if (statusRes.status === 'completed') this.onTaskComplete(statusRes);
                    this.setButtonsDisabled(false);
                    return;
                }
            } catch { }
        }
        this.setButtonsDisabled(false);
    }

    onTaskComplete(data) {
        const workspaceDir = this.getActiveWorkspaceDir();
        const pointcloudSource = this.currentTaskDepthResultId;
        if (pointcloudSource) {
            this.appendLog(`\n${pointcloudSource} 点云任务完成`);
            this.currentTaskDepthResultId = null;
            this.currentTaskId = null;
            if (this.getSelectedDepthResultId() === pointcloudSource) {
                this.loadPointCloudState(pointcloudSource, true);
            }
            this.setButtonsDisabled(false);
            return;
        }
        this.appendLog('\n管道执行完成！');

        // 记录完成的动作
        const action = data.action || 'full';
        if (!this._completedActions) this._completedActions = new Set();
        this._completedActions.add(action);

        // COLMAP 相关操作完成后启用相机可视化按钮
        const poseActions = ['pose', 'full', 'incremental_mapping'];
        if (poseActions.includes(action)) {
            this.elements.showCamerasBtn.disabled = false;
            this.elements.showSparseBtn.disabled = false;
        }

        if (data.result_ply) {
            this.appendLog(`点云文件: ${data.result_ply}`);
            this.updatePlyInfo(data.result_ply);
            this.elements.downloadBtn.disabled = false;
            this.elements.toggleCamerasBtn.disabled = false;
            this.loadPly(workspaceDir);
        }
        // 保存结果和完成状态到项目
        this.saveProject({
            result_ply: data.result_ply || '',
            completed_actions: Array.from(this._completedActions),
        });
        this.setButtonsDisabled(false);
        this.currentTaskId = null;
        // 刷新深度估计结果列表
        this.loadDepthResults();
    }

    onTaskFailed(data) {
        this.appendLog(`\n管道执行失败: ${data.error || '未知错误'}`);
        const pointcloudSource = this.currentTaskDepthResultId;
        this.currentTaskDepthResultId = null;
        this.currentTaskId = null;
        if (pointcloudSource && this.getSelectedDepthResultId() === pointcloudSource) {
            this.loadPointCloudState(pointcloudSource, false);
        }
        this.setButtonsDisabled(false);
    }

    async cancelTask() {
        if (!this.currentTaskId) return;

        try {
            await this.api.cancelTask(this.currentTaskId);
            this.appendLog('\n正在取消任务...');
        } catch (err) {
            this.appendLog(`取消失败: ${err.message}`);
        }
    }

    setStatus(status, label) {
        const badge = this.elements.statusBadge;
        badge.className = `status-badge ${status}`;
        badge.textContent = label;
    }

    appendLog(message) {
        this.logLines.push(message);
        if (this.logLines.length > 1000) {
            this.logLines = this.logLines.slice(-500);
        }
        this.elements.logs.textContent = this.logLines.join('\n');
        this.elements.logs.scrollTop = this.elements.logs.scrollHeight;
        // 同步到 _logLines（项目保存用）
        this._logLines = this.logLines;
        // 检测阶段并更新进度条
        this._detectAndUpdateProgress(message);
    }

    // ====================================================================
    // 进度条
    // ====================================================================
    _detectAndUpdateProgress(logMsg) {
        const stageMap = [
            { key: 'upload',   patterns: ['上传完成'] },
            { key: 'extract',  patterns: ['feature extraction', 'Found .* images in', 'Running pycolmap feature'] },
            { key: 'match',    patterns: ['exhaustive matching', 'feature matching'] },
            { key: 'pose',     patterns: ['incremental mapping', 'Running pycolmap incremental'] },
            { key: 'convert',  patterns: ['Undistorting', 'Converting COLMAP', 'format convert'] },
            { key: 'depth',    patterns: ['Running depth estimation', 'run_pipeline.*depth', 'depth estimation'] },
            { key: 'fusion',   patterns: ['Running fusion', '点云融合', 'fusion'] },
        ];
        // 上传图片的进度单独处理
        if (logMsg.includes('上传进度')) {
            this._currentProgressStage = 0;
            this._showProgress();
            return;
        }
        for (let i = 0; i < stageMap.length; i++) {
            const stage = stageMap[i];
            for (const pat of stage.patterns) {
                if (new RegExp(pat, 'i').test(logMsg)) {
                    this._currentProgressStage = i;
                    this._showProgress();
                    return;
                }
            }
        }
        // 管道完成
        if (logMsg.includes('管道执行完成') || logMsg.includes('Pipeline finished')) {
            this._currentProgressStage = 7;
            this._showProgress();
        }
    }

    _showProgress() {
        const container = document.getElementById('progress_bar_container');
        const fill = document.getElementById('progress_bar_fill');
        const stages = document.querySelectorAll('.progress-stage');
        if (!container || !fill) return;

        // 始终显示进度条
        container.classList.remove('hidden');
        const total = 7;
        const current = Math.min(this._currentProgressStage, total);
        const pct = Math.round((current / total) * 100);
        fill.style.width = pct + '%';

        stages.forEach((el, i) => {
            el.classList.remove('active', 'done');
            if (i < current) el.classList.add('done');
            else if (i === current && current < total) el.classList.add('active');
        });
    }

    resetProgress() {
        this._currentProgressStage = 0;
        const fill = document.getElementById('progress_bar_fill');
        const stages = document.querySelectorAll('.progress-stage');
        if (fill) fill.style.width = '0%';
        stages.forEach(el => el.classList.remove('active', 'done'));
    }

    clearLogs() {
        this.logLines = [];
        this._logLines = [];
        this.elements.logs.textContent = '';
        this.resetProgress();
    }

    setButtonsDisabled(disabled) {
        document.querySelectorAll('[data-action]').forEach(btn => {
            btn.disabled = disabled;
        });
        [
            this.elements.btnDeepFusion,
            this.elements.btnVoxel,
            this.elements.btnStatFilter,
            this.elements.btnRadiusFilter,
        ].forEach(btn => {
            if (btn) btn.disabled = disabled;
        });
        this.elements.cancelBtn.disabled = !disabled;
    }

    updatePlyInfo(path) {
        this.elements.plyInfo.textContent = path.split('\\').pop() || path;
        this.elements.plyInfo.title = path;
    }

    downloadPly() {
        const depthResultId = this.getSelectedDepthResultId();
        const pointcloudPath = this.getSelectedPointCloud();
        let url;
        if (depthResultId && pointcloudPath) {
            url = this.api.getDepthResultPointcloudUrl(
                this.getActiveWorkspaceDir(),
                depthResultId,
                pointcloudPath,
            );
            if (this.api.token) {
                url += `${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(this.api.token)}`;
            }
        } else {
            url = this.api.getPlyUrl('');
        }
        window.open(url, '_blank');
    }

    toggleCameras() {
        const btn = this.elements.toggleCamerasBtn;
        if (!this.viewer) return;
        this.camerasVisible = !this.camerasVisible;
        this.viewer.toggleCameras(this.camerasVisible);
        btn.textContent = this.camerasVisible ? '隐藏相机' : '显示相机';
    }

    async showCameras() {
        if (!this.viewer) {
            this.appendLog('请先加载点云');
            return;
        }
        const workspaceDir = this.getActiveWorkspaceDir();
        const sceneName = this.getActiveSceneName();
        this.appendLog(`正在加载相机位姿 (${workspaceDir}/${sceneName}) ...`);
        try {
            const camerasUrl = this.api.getCamerasUrl(workspaceDir, sceneName);
            await this.viewer.loadCameras(camerasUrl, this.api.token);
            this.camerasVisible = true;
            this.viewer.toggleCameras(true);
            this.elements.toggleCamerasBtn.disabled = false;
            this.elements.toggleCamerasBtn.textContent = '隐藏相机';
            this.appendLog('相机位姿加载完成');
        } catch (err) {
            this.appendLog(`加载相机位姿失败: ${err.message}`);
        }
    }

    async showSparsePly() {
        if (!this.viewer) {
            this.appendLog('请先加载点云');
            return;
        }
        const workspaceDir = this.getActiveWorkspaceDir();
        const sceneName = this.getActiveSceneName();
        this.appendLog(`正在加载稀疏点云 (${workspaceDir}/${sceneName}) ...`);
        try {
            const plyUrl = this.api.getSparsePlyUrl(workspaceDir, sceneName);
            await this.viewer.load(plyUrl, this.api.token);
            this.appendLog('稀疏点云加载完成');
        } catch (err) {
            this.appendLog(`加载稀疏点云失败: ${err.message}`);
        }
    }

    // ====================================================================
    // 深度估计来源管理
    // ====================================================================
    async loadDepthResults() {
        const workspaceDir = this.getActiveWorkspaceDir();
        try {
            const data = await this.api.getDepthResults(workspaceDir);
            const sel = this.elements.depthSource;
            if (!sel) return;
            const currentVal = sel.value || this._preferredDepthResultId;
            sel.innerHTML = '<option value="">-- 选择深度估计结果 --</option>';
            (data.results || []).forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.id;
                opt.dataset.supportsFusion = r.supports_fusion ? 'true' : 'false';
                const extra = r.has_ply ? ' [含点云]' : '';
                opt.textContent = `${r.method} (${r.depth_count} 张深度图)${extra}`;
                sel.appendChild(opt);
            });
            const availableValues = [...sel.options].map(option => option.value).filter(Boolean);
            sel.value = availableValues.includes(currentVal) ? currentVal : (availableValues[0] || '');
            this._preferredDepthResultId = sel.value;
            await this.loadPointCloudState(sel.value, true);
        } catch (err) {
            this.appendLog(`加载深度估计结果失败: ${err.message}`);
        }
    }

    async loadPointCloudState(depthResultId, previewSelected = false) {
        const input = this.elements.pointcloudInput;
        const tasks = this.elements.pointcloudTasks;
        if (!input || !tasks) return;
        if (!depthResultId) {
            input.innerHTML = '<option value="">-- 暂无可用点云 --</option>';
            input.disabled = true;
            this.renderPointCloudTasks([]);
            return;
        }

        const workspaceDir = this.getActiveWorkspaceDir();
        const previousInput = input.value;
        try {
            const state = await this.api.getDepthResultPointclouds(workspaceDir, depthResultId);
            if (this.elements.depthSource?.value !== depthResultId) return;

            input.replaceChildren();
            const pointclouds = state.pointclouds || [];
            if (pointclouds.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = '-- 请先执行点云融合 --';
                input.appendChild(option);
                input.disabled = true;
            } else {
                pointclouds.forEach(pointcloud => {
                    const option = document.createElement('option');
                    option.value = pointcloud.path;
                    const category = pointcloud.category === 'processed' ? '处理' : '融合';
                    option.textContent = `[${category}] ${pointcloud.name}`;
                    input.appendChild(option);
                });
                input.disabled = false;
                if (pointclouds.some(pointcloud => pointcloud.path === previousInput)) {
                    input.value = previousInput;
                }
            }
            this.renderPointCloudTasks(state.tasks || []);
            if (previewSelected && input.value) {
                await this.loadSelectedPointCloud();
            }
        } catch (err) {
            input.innerHTML = '<option value="">-- 加载失败 --</option>';
            input.disabled = true;
            this.renderPointCloudTasks([]);
            this.appendLog(`加载点云任务失败: ${err.message}`);
        }
    }

    renderPointCloudTasks(taskRecords) {
        const container = this.elements.pointcloudTasks;
        if (!container) return;
        container.replaceChildren();
        if (!taskRecords.length) {
            const empty = document.createElement('p');
            empty.className = 'pointcloud-task-empty';
            empty.textContent = '当前深度结果暂无点云处理任务';
            container.appendChild(empty);
            return;
        }

        const statusLabels = {
            pending: '等待中',
            running: '运行中',
            completed: '已完成',
            failed: '失败',
        };
        taskRecords.forEach(record => {
            const row = document.createElement('div');
            row.className = 'pointcloud-task-row';

            const method = document.createElement('span');
            method.className = 'pointcloud-task-method';
            method.textContent = record.method || 'unknown';

            const detail = document.createElement('span');
            detail.className = 'pointcloud-task-detail';
            detail.textContent = record.output_ply || record.input_ply || '深度图';
            detail.title = record.error || detail.textContent;

            const status = document.createElement('span');
            status.className = `pointcloud-task-status ${record.status || 'pending'}`;
            status.textContent = statusLabels[record.status] || record.status || '等待中';

            row.append(method, detail, status);
            container.appendChild(row);
        });
    }

    async loadSelectedPointCloud() {
        const depthResultId = this.getSelectedDepthResultId();
        const pointcloudPath = this.getSelectedPointCloud();
        if (!depthResultId || !pointcloudPath || !this.viewer) return;
        const workspaceDir = this.getActiveWorkspaceDir();
        const url = this.api.getDepthResultPointcloudUrl(
            workspaceDir,
            depthResultId,
            pointcloudPath,
        );
        try {
            await this.viewer.load(url, this.api.token);
            this.updatePlyInfo(pointcloudPath);
            this.elements.downloadBtn.disabled = false;
            this.elements.toggleCamerasBtn.disabled = false;
        } catch (err) {
            this.appendLog(`加载所选点云失败: ${err.message}`);
        }
    }

    getSelectedDepthResultId() {
        const sel = this.elements.depthSource;
        return sel ? sel.value : '';
    }

    getSelectedPointCloud() {
        const sel = this.elements.pointcloudInput;
        return sel ? sel.value : '';
    }

    async runDeepFusion() {
        const workspaceDir = this.getActiveWorkspaceDir();
        const sceneName = this.getActiveSceneName();
        const depthResultId = this.getSelectedDepthResultId();
        if (!depthResultId) {
            this.appendLog('请先选择深度估计结果');
            return;
        }
        const selectedOption = this.elements.depthSource?.selectedOptions?.[0];
        if (selectedOption?.dataset?.supportsFusion !== 'true') {
            this.appendLog(`当前深度结果不支持点云融合: ${depthResultId}`);
            return;
        }
        const conf = parseFloat(this.elements.deepFusionConf.value) || 0.9;
        const thresView = parseInt(this.elements.deepFusionThres.value) || 5;
        const numWorker = parseInt(this.elements.fusionNumWorker?.value, 10) || 4;

        this.appendLog(`启动 ${depthResultId} 点云融合 (conf=${conf}, thres_view=${thresView})`);
        this.setButtonsDisabled(true);
        this.setStatus('running', '运行中');

        try {
            const result = await this.api.runDeepFusion({
                workspace_dir: workspaceDir,
                scene_name: sceneName,
                depth_result_id: depthResultId,
                conf,
                thres_view: thresView,
                num_worker: numWorker,
            });
            this.currentTaskId = result.task_id;
            this.currentTaskDepthResultId = depthResultId;
            this.appendLog(`任务 ID: ${result.task_id}`);
            this.connectWebSocket(result.task_id);
        } catch (err) {
            this.appendLog(`深度融合启动失败: ${err.message}`);
            this.setStatus('failed', '失败');
            this.setButtonsDisabled(false);
        }
    }

    async runPointCloudProcess(method) {
        const workspaceDir = this.getActiveWorkspaceDir();
        const sceneName = this.getActiveSceneName();
        const depthResultId = this.getSelectedDepthResultId();
        const inputPly = this.getSelectedPointCloud();
        if (!depthResultId) {
            this.appendLog('请先选择深度估计结果');
            return;
        }
        if (!inputPly) {
            this.appendLog('请先选择输入点云');
            return;
        }
        const params = {
            method,
            workspace_dir: workspaceDir,
            scene_name: sceneName,
            depth_result_id: depthResultId,
            input_ply: inputPly,
        };

        if (method === 'voxel') {
            params.voxel_size = parseFloat(this.elements.voxelSize?.value) || 0.01;
            this.appendLog(`启动体素下采样 (voxel_size=${params.voxel_size})`);
        } else if (method === 'statistical') {
            params.nb_neighbors = parseInt(this.elements.statNbNeighbors?.value, 10) || 20;
            params.std_ratio = parseFloat(this.elements.statStdRatio?.value) || 2.0;
            this.appendLog(`启动统计滤波 (nb_neighbors=${params.nb_neighbors}, std_ratio=${params.std_ratio})`);
        } else if (method === 'radius') {
            params.nb_points = parseInt(this.elements.radiusNbPoints?.value, 10) || 16;
            params.radius = parseFloat(this.elements.radiusValue?.value) || 0.05;
            this.appendLog(`启动半径滤波 (nb_points=${params.nb_points}, radius=${params.radius})`);
        } else {
            this.appendLog(`未知点云处理方法: ${method}`);
            return;
        }

        this.setButtonsDisabled(true);
        this.setStatus('running', '运行中');

        try {
            const result = await this.api.runPointCloudProcess(params);
            this.currentTaskId = result.task_id;
            this.currentTaskDepthResultId = depthResultId;
            this.appendLog(`任务 ID: ${result.task_id}`);
            this.connectWebSocket(result.task_id);
        } catch (err) {
            this.appendLog(`点云处理启动失败: ${err.message}`);
            this.setStatus('failed', '失败');
            this.setButtonsDisabled(false);
        }
    }

    // ====================================================================
    // MVS 方法和 checkpoint 管理
    // ====================================================================
    async loadMvsMethods() {
        try {
            const data = await this.api.getMvsMethods();
            const sel = this.elements.mvsRoot;
            if (!sel) return;
            const currentVal = sel.value;
            sel.innerHTML = '<option value="">-- 选择 MVS 方法 --</option>';
            (data.methods || []).forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.path;
                opt.dataset.name = m.name;
                opt.textContent = m.name;
                sel.appendChild(opt);
            });
            // 恢复之前的选择，否则默认选中第一个方法
            if (currentVal && [...sel.options].some(o => o.value === currentVal)) {
                sel.value = currentVal;
            } else if (sel.options.length > 1) {
                sel.selectedIndex = 1;
            }
            // 联动加载 checkpoint
            await this.updateCheckpointList();
        } catch (err) {
            console.warn('Failed to load MVS methods:', err);
        }
    }

    async updateCheckpointList(preferredPath = '') {
        const methodPath = this.elements.mvsRoot ? this.elements.mvsRoot.value : '';
        const sel = this.elements.checkpoint;
        if (!sel) return;
        const previous = preferredPath || sel.value;
        sel.innerHTML = '<option value="">-- 选择 checkpoint --</option>';

        if (!methodPath) return;

        // 从路径中提取方法名（兼容 Windows 反斜杠）
        const methodName = methodPath.split(/[/\\]/).filter(Boolean).pop();
        try {
            const data = await this.api.getMvsCheckpoints(methodName);
            (data.checkpoints || []).forEach(cp => {
                const opt = document.createElement('option');
                opt.value = cp.path;
                opt.textContent = cp.name;
                sel.appendChild(opt);
            });
            // 恢复选择，否则默认选中第一个
            if (previous && [...sel.options].some(o => o.value === previous)) {
                sel.value = previous;
            } else if (sel.options.length > 1) {
                sel.selectedIndex = 1;
            }
        } catch (err) {
            console.warn('Failed to load checkpoints:', err);
        }
    }

    async _restoreMvsSelection(mvsRoot, checkpoint) {
        // 确保方法列表已加载
        if (!this.elements.mvsRoot || this.elements.mvsRoot.options.length <= 1) {
            await this.loadMvsMethods();
        }
        if (mvsRoot && this.elements.mvsRoot) {
            // 兼容路径/名称两种存储格式
            const options = [...this.elements.mvsRoot.options];
            const matched = options.find(o => o.value === mvsRoot || o.textContent === mvsRoot || o.dataset.name === mvsRoot);
            if (matched) {
                this.elements.mvsRoot.value = matched.value;
            }
        }
        await this.updateCheckpointList(checkpoint || '');
    }

    switchTab(tab) {
        // 切换标签按钮状态
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelector(`.tab-btn[data-tab="${tab}"]`).classList.add('active');
        // 切换内容面板
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`tab_${tab}`).classList.add('active');
        if (tab === 'pointcloud') {
            this.loadDepthResults();
        }
    }

    setTheme(theme) {
        // 设置 html 和 body 元素的 data-theme 属性
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        // 更新按钮状态
        document.querySelectorAll('.theme-btn').forEach(btn => {
            if (btn.dataset.theme === theme) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        console.log('Theme set to:', theme);
    }

    loadTheme() {
        const saved = localStorage.getItem('theme') || 'dark';
        this.setTheme(saved);
    }

    syncSharedField(name) {
        const id1 = `colmap_${name}`;
        const id2 = `mvs_${name}`;
        const el1 = document.getElementById(id1);
        const el2 = document.getElementById(id2);
        if (!el1 || !el2) return;
        // 同步工作目录和场景名称
        const activeTab = document.querySelector('.tab-btn.active')?.dataset?.tab;
        if (activeTab === 'colmap') {
            el2.value = el1.value;
        } else {
            el1.value = el2.value;
        }
    }

    initViewer() {
        const container = this.elements.threeContainer;
        this.viewer = new PlyViewer(container);
        // 加载保存的主题
        this.loadTheme();
        // 加载 MVS 方法列表
        this.loadMvsMethods();
        // 尝试自动登录
        this.tryAutoLogin();
    }

    async tryAutoLogin() {
        // 1. 先尝试本地 token
        if (this.api.token) {
            const me = await this.api.getMe();
            if (me && me.username) {
                this.setLoggedIn(me.username);
                this.appendLog(`已自动登录为 ${me.username}`);
                return;
            }
            this.api.logout();
        }

        // 2. 尝试开发账号自动登录
        this.elements.loginHint.textContent = '正在自动登录...';
        const result = await this.api.login('dev', 'dev');
        if (result.ok) {
            this.setLoggedIn(result.username);
            this.appendLog(`已自动登录为 ${result.username}（开发账号）`);
        } else {
            this.elements.loginHint.textContent = '';
        }
    }

    setLoggedIn(username) {
        this.isLoggedIn = true;
        this.elements.loginOverlay.style.display = 'none';
        this.elements.appContainer.style.display = 'flex';
        this.elements.authArea.style.display = 'flex';
        this.elements.authUsername.textContent = username;
        this.elements.projectArea.style.display = 'flex';
        this.appendLog(`--- 当前用户: ${username} ---`);
        // 加载项目列表
        this.loadProjects();
    }

    setLoggedOut() {
        this.isLoggedIn = false;
        this.elements.loginOverlay.style.display = 'flex';
        this.elements.appContainer.style.display = 'none';
        this.elements.authArea.style.display = 'none';
        this.elements.projectArea.style.display = 'none';
        this.elements.loginUsername.value = '';
        this.elements.loginPassword.value = '';
        this.elements.loginError.textContent = '';
        this.elements.loginHint.textContent = '';
        this.currentProjectId = null;
    }

    onAuthExpired() {
        this.setLoggedOut();
        this.elements.loginError.textContent = '登录已过期，请重新登录';
    }

    async login() {
        const username = this.elements.loginUsername.value.trim();
        const password = this.elements.loginPassword.value;
        this.elements.loginError.textContent = '';
        if (!username || !password) {
            this.elements.loginError.textContent = '请输入用户名和密码';
            return;
        }
        const result = await this.api.login(username, password);
        if (result.ok) {
            this.setLoggedIn(result.username);
        } else {
            this.elements.loginError.textContent = result.detail || '登录失败';
        }
    }

    async register() {
        const username = this.elements.loginUsername.value.trim();
        const password = this.elements.loginPassword.value;
        this.elements.loginError.textContent = '';
        if (!username || !password) {
            this.elements.loginError.textContent = '请输入用户名和密码';
            return;
        }
        const result = await this.api.register(username, password);
        if (result.ok) {
            this.setLoggedIn(result.username);
        } else {
            this.elements.loginError.textContent = result.detail || '注册失败';
        }
    }

    logout() {
        this.api.logout();
        this.setLoggedOut();
        this.elements.loginHint.textContent = '已退出登录';
    }

    async loadPly(workspaceDir) {
        const url = this.api.getPlyUrl(workspaceDir || '');
        try {
            await this.viewer.load(url, this.api.token);
            // 加载相机位姿
            const sceneName = this.getActiveSceneName();
            const camerasUrl = this.api.getCamerasUrl(workspaceDir || '', sceneName);
            try {
                await this.viewer.loadCameras(camerasUrl, this.api.token);
                this.camerasVisible = true;
                this.viewer.toggleCameras(true);
                this.elements.toggleCamerasBtn.textContent = '隐藏相机';
                this.elements.showCamerasBtn.disabled = false;
                this.elements.showSparseBtn.disabled = false;
            } catch (camErr) {
                this.appendLog(`相机位姿加载失败: ${camErr.message}`);
            }
        } catch (err) {
            this.appendLog(`加载点云失败: ${err.message}`);
        }
    }

    // ====================================================================
    // 项目管理
    // ====================================================================

    async loadProjects() {
        const res = await this.api.getProjects();
        const sel = this.elements.projectSelector;
        sel.innerHTML = '<option value="">-- 选择项目 --</option>';
        if (res.projects && res.projects.length > 0) {
            res.projects.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.id;
                // 显示名称 + 更新时间
                const date = p.updated_at ? p.updated_at.slice(0, 16).replace('T', ' ') : '';
                opt.textContent = `${p.name} (${date})`;
                sel.appendChild(opt);
            });
            // 尝试加载最近的项目（或已有 currentProjectId）
            const targetId = this.currentProjectId || res.projects[0].id;
            sel.value = targetId;
            await this.selectProject(targetId);
        }
    }

    async selectProject(projectId) {
        const data = await this.api.getProject(projectId);
        if (!data) {
            this.appendLog('项目加载失败');
            return;
        }
        // 切换项目前重置界面状态
        this._resetProjectUI();
        this.currentProjectId = projectId;
        this.elements.projectSelector.value = projectId;
        this.elements.deleteProjectBtn.disabled = false;  // 启用删除按钮
        this.loadProjectIntoUI(data);
        this.appendLog(`已加载项目: ${data.name}`);
        // 加载深度估计结果列表
        this.loadDepthResults();
    }

    _resetProjectUI() {
        // 重置日志
        this.logLines = [];
        this._logLines = [];
        this.elements.logs.textContent = '';

        // 重置进度条
        this.resetProgress();

        // 重置 3D 视图
        if (this.viewer) {
            this.viewer.removeCameras();
            this.camerasVisible = false;
            this.elements.toggleCamerasBtn.textContent = '相机位姿';
            this.elements.toggleCamerasBtn.disabled = true;
        }

        // 重置图像画廊
        const gallery = document.getElementById('image_gallery');
        const scroll = document.getElementById('gallery_scroll');
        if (gallery) gallery.style.display = 'none';
        if (scroll) scroll.innerHTML = '';

        // 重置上传状态
        const uploadGroup = document.getElementById('upload_group');
        if (uploadGroup) {
            uploadGroup.innerHTML = `<label>上传图像</label><input type="file" id="image_upload" webkitdirectory multiple>`;
            // 重新绑定上传事件
            document.getElementById('image_upload').addEventListener('change', async (e) => {
                await this.uploadImages(e.target.files);
            });
        }

        // 重置按钮状态
        this.elements.showCamerasBtn.disabled = true;
        this.elements.showSparseBtn.disabled = true;
        this.elements.downloadBtn.disabled = true;
        this.elements.plyInfo.textContent = '无数据';

        // 重置完成动作
        this._completedActions = new Set();
        this._preferredDepthResultId = '';
        this.currentTaskDepthResultId = null;
        if (this.elements.depthSource) {
            this.elements.depthSource.innerHTML = '<option value="">-- 选择深度估计结果 --</option>';
        }
        if (this.elements.pointcloudInput) {
            this.elements.pointcloudInput.innerHTML = '<option value="">-- 暂无可用点云 --</option>';
            this.elements.pointcloudInput.disabled = true;
        }
        this.renderPointCloudTasks([]);
    }

    loadProjectIntoUI(data) {
        // 恢复配置
        if (data.config) {
            const cfg = data.config;
            const setVal = (id, val) => {
                const el = document.getElementById(id);
                if (el && val !== undefined && val !== null && val !== '') {
                    if (el.type === 'checkbox') {
                        el.checked = !!val;
                    } else {
                        el.value = val;
                    }
                }
            };
            // 工作目录：始终设置（项目专属路径 > 配置值 > 默认值）
            setVal('colmap_workspace_dir', cfg.workspace_dir || `data/${this.getActiveUsername()}/workspace/${data.id}`);
            setVal('mvs_workspace_dir', cfg.workspace_dir || `data/${this.getActiveUsername()}/workspace/${data.id}`);
            setVal('image_dir', cfg.image_dir || data.image_upload_dir);
            setVal('interval_scale', cfg.interval_scale);
            setVal('num_view', cfg.num_view);
            setVal('max_h', cfg.max_h);
            setVal('max_w', cfg.max_w);
            setVal('conf', cfg.conf);
            setVal('thres_view', cfg.thres_view);
            setVal('num_worker', cfg.num_worker);
            setVal('filter_method', cfg.filter_method);
            setVal('camera_model', cfg.camera_model);
            setVal('single_camera', cfg.single_camera);
            setVal('use_gpu', cfg.use_gpu);
            setVal('sfm_max_image_size', cfg.sfm_max_image_size);
            setVal('sfm_max_num_features', cfg.sfm_max_num_features);
            this._preferredDepthResultId = cfg.depth_result_id || '';

            // MVS 方法 / checkpoint 需要先填充下拉选项再赋值
            this._restoreMvsSelection(cfg.mvs_root, cfg.checkpoint);
        } else if (data.image_upload_dir) {
            // 至少恢复上传目录
            document.getElementById('image_dir').value = data.image_upload_dir;
        }

        // 恢复图像预览画廊
        const uploadDir = (data.config && data.config.image_dir) || data.image_upload_dir;
        if (uploadDir) {
            this.loadImageGallery(uploadDir);
            // 恢复上传状态显示
            const uploadGroup = document.getElementById('upload_group');
            if (uploadGroup) {
                uploadGroup.innerHTML = `<label>图像状态</label><span class="upload-status">已上传</span>`;
            }
        }

        // 恢复日志
        if (data.logs && data.logs.length > 0) {
            this._logLines = data.logs;
            this.elements.logs.textContent = this._logLines.join('\n');
            this.elements.logs.scrollTop = this.elements.logs.scrollHeight;
        }

        // 恢复结果
        if (data.result_ply) {
            this.updatePlyInfo(data.result_ply);
            this.elements.downloadBtn.disabled = false;
            this.elements.toggleCamerasBtn.disabled = false;
            this.loadPly(this.getActiveWorkspaceDir());
        }

        // 恢复已完成的管道动作状态
        if (data.completed_actions && data.completed_actions.length > 0) {
            this._completedActions = new Set(data.completed_actions);
        } else {
            this._completedActions = new Set();
        }
        // 如果 pose 或 full 已完成，或存在 result_ply（旧项目兼容），启用相机可视化按钮
        if (this._completedActions.has('pose') || this._completedActions.has('full') || data.result_ply) {
            this.elements.showCamerasBtn.disabled = false;
            this.elements.showSparseBtn.disabled = false;
        }

        // 恢复进度条状态
        if (this._completedActions.size > 0) {
            this._restoreProgressFromActions(this._completedActions);
        }

        // 从服务器检查实际文件状态，更新进度
        this._checkAndUpdatePipelineStatus();
    }

    async _checkAndUpdatePipelineStatus() {
        const workspaceDir = this.getActiveWorkspaceDir();
        const sceneName = this.getActiveSceneName();
        try {
            const status = await this.api.getPipelineStatus(workspaceDir, sceneName);

            // 根据文件存在状态更新进度条
            const stageOrder = [
                { key: 'feature_extract', stage: 1 },
                { key: 'feature_match', stage: 2 },
                { key: 'incremental_mapping', stage: 3 },
                { key: 'convert', stage: 4 },
                { key: 'fusion', stage: 6 },
            ];

            let maxStage = 0;
            for (const item of stageOrder) {
                if (status[item.key]) {
                    maxStage = Math.max(maxStage, item.stage);
                }
            }

            if (maxStage > 0) {
                this._currentProgressStage = maxStage;
                this._showProgress();
            }

            // 位姿估计完成后启用相机和稀疏点云按钮
            if (status.incremental_mapping) {
                this.elements.showCamerasBtn.disabled = false;
                this.elements.showSparseBtn.disabled = false;
                if (!this._completedActions) this._completedActions = new Set();
                this._completedActions.add('pose');
            }

            // 点云融合完成后启用下载按钮
            if (status.fusion) {
                this.elements.downloadBtn.disabled = false;
                this.elements.toggleCamerasBtn.disabled = false;
            }
        } catch (err) {
            console.warn('Pipeline status check failed:', err);
        }
    }

    _restoreProgressFromActions(actions) {
        const stageMap = {
            'feature_extract': 1,
            'feature_match': 2,
            'incremental_mapping': 3,
            'pose': 3,
            'convert': 4,
            'depth': 5,
            'fusion': 6,
        };
        let maxStage = 0;
        for (const action of actions) {
            const stage = stageMap[action];
            if (stage !== undefined && stage > maxStage) {
                maxStage = stage;
            }
        }
        if (maxStage > 0) {
            this._currentProgressStage = maxStage;
            this._showProgress();
        }
    }

    async createProject() {
        const name = prompt('请输入项目名称（可选）:', '');
        if (name === null) return;  // 取消
        const data = await this.api.createProject(name || '');
        if (data.id) {
            this.currentProjectId = data.id;
            this.appendLog(`已创建新项目: ${data.name} (${data.id})`);
            // 清空日志和结果
            this._logLines = [];
            this.clearLogs();
            this.elements.plyInfo.textContent = '无点云';
            this.elements.downloadBtn.disabled = true;
            this.elements.toggleCamerasBtn.disabled = true;
            // 刷新项目列表
            await this.loadProjects();
        }
    }

    async deleteProject() {
        if (!this.currentProjectId) {
            this.appendLog('请先选择要删除的项目');
            return;
        }

        // 弹出确认对话框
        const confirmDelete = confirm('确定要删除当前项目吗？');
        if (!confirmDelete) return;

        const confirmFiles = confirm('是否同时删除本地工作目录文件？\n\n选择"确定"删除所有文件\n选择"取消"仅删除项目配置');

        try {
            const result = await this.api.deleteProject(this.currentProjectId, confirmFiles);
            this.appendLog(`项目已删除${result.files_deleted ? '（含本地文件）' : ''}`);

            // 重置界面
            this.currentProjectId = null;
            this._resetProjectUI();

            // 刷新项目列表
            await this.loadProjects();
        } catch (err) {
            this.appendLog(`删除失败: ${err.message}`);
        }
    }

    async saveProject(partial) {
        if (!this.currentProjectId) return;
        // 收集当前配置
        const config = this.getConfig();
        const data = {
            config,
            logs: this._logLines,
            ...partial,
        };
        await this.api.updateProject(this.currentProjectId, data);
    }

    scheduleSaveProject() {
        if (this._saveTimer) clearTimeout(this._saveTimer);
        this._saveTimer = setTimeout(() => {
            this.saveProject({});
        }, 2000);
    }

    /** 获取当前激活标签的工作目录 */
    getActiveWorkspaceDir() {
        const activeTab = document.querySelector('.tab-btn.active')?.dataset?.tab || 'colmap';
        const id = activeTab === 'colmap' ? 'colmap_workspace_dir' : 'mvs_workspace_dir';
        return document.getElementById(id).value || 'runs';
    }

    /** 获取当前场景名称（固定值） */
    getActiveSceneName() {
        return 'scene';
    }

    /** 获取当前登录用户名 */
    getActiveUsername() {
        const el = document.getElementById('auth_username_display');
        return el ? el.textContent : 'dev';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new MvsApp();
});
