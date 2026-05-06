document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const dropArea = document.getElementById('file-drop-area');
    const fileInput = document.getElementById('data-file');
    const uploadForm = document.getElementById('upload-form');
    const uploadBtnText = uploadForm.querySelector('.btn-text');
    const uploadLoader = uploadForm.querySelector('.loader');
    const uploadStatus = document.getElementById('upload-status');
    const fileMsg = dropArea.querySelector('.file-msg');
    
    const startTrainingBtn = document.getElementById('start-training-btn');
    const trainingBtnText = startTrainingBtn.querySelector('.btn-text');
    const trainingLoader = startTrainingBtn.querySelector('.loader');
    const statusBadge = document.getElementById('training-status-text');
    
    const terminal = document.getElementById('terminal-output');
    
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const chatHistory = document.getElementById('chat-history');

    let isTraining = false;
    let statusInterval = null;

    // --- Drag and Drop Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if(files.length) {
            fileInput.files = files;
            updateFileMsg(files[0].name);
        }
    });

    fileInput.addEventListener('change', function() {
        if(this.files.length) {
            updateFileMsg(this.files[0].name);
        }
    });

    function updateFileMsg(name) {
        fileMsg.textContent = name;
        uploadStatus.className = 'status-msg';
        uploadStatus.textContent = '';
    }

    // --- Upload Dataset ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if(!fileInput.files.length) return;

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        setLoading(uploadBtnText, uploadLoader, true, "Uploading...");
        uploadStatus.className = 'status-msg';
        uploadStatus.textContent = '';

        try {
            const res = await fetch('/api/upload_data', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                uploadStatus.textContent = 'Dataset uploaded successfully!';
                uploadStatus.classList.add('success');
                startTrainingBtn.disabled = false; // Enable training
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.textContent = error.message;
            uploadStatus.classList.add('error');
        } finally {
            setLoading(uploadBtnText, uploadLoader, false, "Upload Dataset");
        }
    });

    // --- Start Training ---
    startTrainingBtn.addEventListener('click', async () => {
        if(isTraining) return;

        setLoading(trainingBtnText, trainingLoader, true, "Starting...");
        
        try {
            const res = await fetch('/api/start_training', {
                method: 'POST'
            });
            const data = await res.json();
            
            if (res.ok) {
                isTraining = true;
                startTrainingBtn.disabled = true;
                updateBadge('running');
                terminal.innerHTML = ''; // Clear terminal
                
                // Start polling status
                statusInterval = setInterval(pollTrainingStatus, 2000);
            } else {
                throw new Error(data.detail || 'Failed to start training');
            }
        } catch (error) {
            appendTerminal(`Error: ${error.message}`, true);
            setLoading(trainingBtnText, trainingLoader, false, "Start Training");
        }
    });

    async function pollTrainingStatus() {
        try {
            const res = await fetch('/api/training_status');
            const data = await res.json();
            
            // Update terminal logs
            updateTerminal(data.logs);
            
            if (data.status === 'completed' || data.status === 'error') {
                clearInterval(statusInterval);
                isTraining = false;
                
                if (data.status === 'completed') {
                    updateBadge('completed');
                    appendTerminal('Model training completed! Reloading global model for inference...', false, true);
                    await reloadModel();
                } else {
                    updateBadge('error');
                }
                
                setLoading(trainingBtnText, trainingLoader, false, "Start Training");
                startTrainingBtn.disabled = false; // Allow restart
            } else if (data.status === 'running') {
                updateBadge('running');
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }

    async function reloadModel() {
        try {
            const res = await fetch('/api/reload_model', { method: 'POST' });
            if (res.ok) {
                appendTerminal('Model reloaded successfully. Ready for inference.', false, true);
            }
        } catch (error) {
            appendTerminal(`Error reloading model: ${error.message}`, true);
        }
    }

    // --- Inference ---
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if(!query) return;

        // Add user msg to chat
        addChatMsg(query, 'user');
        queryInput.value = '';
        
        // Add temp loading msg
        const loadingId = addChatMsg('Thinking...', 'assistant', true);

        try {
            const res = await fetch('/api/medical_query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: query })
            });
            const data = await res.json();
            
            // Remove loading msg
            document.getElementById(loadingId).remove();
            
            if (res.ok) {
                let formattedRes = data.response;
                if(data.risk_level && data.risk_level !== "Unknown") {
                    formattedRes += `<div class="structured-response">
                        <strong>Risk Level:</strong> ${data.risk_level}<br/>
                        <strong>Possible Tests:</strong> ${data.tests.join(', ')}
                    </div>`;
                }
                addChatMsg(formattedRes, 'assistant', false, true);
            } else {
                throw new Error(data.detail || 'Inference failed');
            }
        } catch (error) {
            document.getElementById(loadingId)?.remove();
            addChatMsg(`Error: ${error.message}`, 'assistant');
        }
    });

    // --- Utility Functions ---
    function setLoading(textEl, loaderEl, isLoading, text) {
        textEl.textContent = text;
        if(isLoading) {
            loaderEl.classList.remove('hidden');
        } else {
            loaderEl.classList.add('hidden');
        }
    }

    function updateBadge(status) {
        statusBadge.className = `badge ${status}`;
        statusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }

    let lastLogCount = 0;
    function updateTerminal(logs) {
        if (!logs || !logs.length) return;
        
        // Only append new logs
        const newLogs = logs.slice(lastLogCount);
        newLogs.forEach(log => appendTerminal(log));
        lastLogCount = logs.length;
    }

    function appendTerminal(text, isError = false, isSuccess = false) {
        const div = document.createElement('div');
        div.className = 'log-line';
        if(isError) div.style.color = 'var(--danger-color)';
        if(isSuccess) div.style.color = 'var(--secondary-color)';
        div.textContent = text;
        terminal.appendChild(div);
        terminal.scrollTop = terminal.scrollHeight;
    }

    function addChatMsg(text, type, isLoading = false, isHTML = false) {
        const id = 'msg-' + Date.now();
        const div = document.createElement('div');
        div.className = `chat-msg ${type}`;
        div.id = id;
        
        const avatarStr = type === 'user' ? 'U' : 'AI';
        
        let contentHtml = '';
        if(isHTML) {
            contentHtml = text;
        } else {
            contentHtml = text.replace(/\\n/g, '<br/>');
        }

        if(isLoading) {
            contentHtml = `<span style="opacity: 0.5;">Thinking <span class="loader hidden" style="width: 10px; height: 10px; display: inline-block; border-color: rgba(255,255,255,0.5); border-top-color: #fff;"></span></span>`;
        }
        
        div.innerHTML = `
            <div class="avatar">${avatarStr}</div>
            <div class="msg-content">${contentHtml}</div>
        `;
        
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        if(isLoading) {
            div.querySelector('.loader').classList.remove('hidden');
        }
        
        return id;
    }
});
