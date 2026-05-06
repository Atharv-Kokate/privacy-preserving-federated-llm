import os
import subprocess
import shutil
import asyncio
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict

router = APIRouter()

# Global state to track subprocesses
training_state = {
    "status": "idle", # idle, running, completed, error
    "server_process": None,
    "client_processes": [],
    "logs": []
}

DATASET_DIR = "datasets"

def read_output(process, prefix):
    """Background thread to read process output and append to logs."""
    for line in iter(process.stdout.readline, ''):
        if line:
            training_state["logs"].append(f"[{prefix}] {line.strip()}")
            # Keep only the last 200 logs to avoid memory issues
            if len(training_state["logs"]) > 200:
                training_state["logs"] = training_state["logs"][-200:]
    process.stdout.close()

@router.post("/api/upload_data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    file_path = os.path.join(DATASET_DIR, "user_uploaded.json")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"message": "Data uploaded successfully.", "filename": "user_uploaded.json"}

@router.post("/api/start_training")
async def start_training():
    global training_state
    
    if training_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Training is already running.")
        
    user_data_path = os.path.join(DATASET_DIR, "user_uploaded.json")
    dummy_data_path = os.path.join(DATASET_DIR, "hospital_A.json")
    
    if not os.path.exists(user_data_path):
        raise HTTPException(status_code=400, detail="Please upload data first.")
    
    # Clear previous state
    training_state["status"] = "running"
    training_state["logs"] = ["Starting Federated Learning Server..."]
    training_state["client_processes"] = []
    
    # Start server
    server_process = subprocess.Popen(
        ["python", "-m", "server.server"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    training_state["server_process"] = server_process
    threading.Thread(target=read_output, args=(server_process, "SERVER"), daemon=True).start()
    
    # Wait briefly for server to start
    await asyncio.sleep(4)
    
    training_state["logs"].append("Starting Clients...")
    
    # Start clients
    client1 = subprocess.Popen(
        ["python", "-m", "client.client", "--server-ip", "127.0.0.1", "--dataset", user_data_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    threading.Thread(target=read_output, args=(client1, "CLIENT-User"), daemon=True).start()
    
    client2 = subprocess.Popen(
        ["python", "-m", "client.client", "--server-ip", "127.0.0.1", "--dataset", dummy_data_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    threading.Thread(target=read_output, args=(client2, "CLIENT-A"), daemon=True).start()
    
    training_state["client_processes"] = [client1, client2]
    
    # Monitor completion in the background
    asyncio.create_task(monitor_training())
    
    return {"message": "Training started successfully."}

async def monitor_training():
    global training_state
    
    # Wait for server to finish
    while True:
        server_poll = training_state["server_process"].poll()
        if server_poll is not None:
            break
        await asyncio.sleep(2)
        
    # Check if there are any errors
    if server_poll != 0:
        training_state["status"] = "error"
        training_state["logs"].append(f"Server exited with code {server_poll}")
    else:
        training_state["status"] = "completed"
        training_state["logs"].append("Federated Learning completed successfully!")
        
        # In a real app, we might automatically trigger a model reload here,
        # but we can also provide a separate endpoint for it or let the inference handler check.

@router.get("/api/training_status")
async def get_training_status():
    global training_state
    return {
        "status": training_state["status"],
        "logs": training_state["logs"]
    }
