from fastapi import FastAPI, UploadFile, File, HTTPException

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import json
import uuid
import os, shutil


app = FastAPI()     # server가 만들어짐..


@app.get('/')
def root1():
    return {
        "message": "Hi!!"
    }

@app.post('/infer')
def infer2(file:UploadFile = File(...)):     # body: form-data
    
    allowed_ext = ["jpg", "jpeg", "png", "bmp", "webp"]
    
    if not file.filename:
        return {"error": "파일명이 없습니다."}
    
    ext = file.filename.split(".")[-1].lower()
    
    if ext not in allowed_ext:
        return {"error": "이미지 파일이 아닙니다! 이미지 파일을 올려주세요."}
    
    # 무조건 데이터는 저장해야 한다. MLops관점에서: 평가 및 재학습
    # 이미지를 uuid로 고유한 이름으로 변경한 뒤 서버가 아닌 클라우드에 저장해야함
    newfile_name = f"{uuid.uuid4()}.{ext}"
    
    file_path = os.path.join("upload_img", newfile_name)
    
    with open(file_path, mode="wb") as buffer:
        shutil.copyfileobj(file.file, buffer)    # file.file은 파일obj, buffer는 새로운 파일obj

    
    #------------------------추론코드-----------------------------
    

    return {
        "result" : "카리나",
        "index" : "2",
        "filename": newfile_name
    }