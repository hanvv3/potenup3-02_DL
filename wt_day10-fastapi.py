from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import os, shutil, io, json, uuid
from PIL import Image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_test = T.Compose(
    [
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔥 startup 영역
    print("============= Model loading initiated =============")
        
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)

    checkpoint = torch.load("./models/celeb_resnet34.pth", map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    app.state.model = model   # ✅ 여기에 저장

    print("================== Model Ready ====================")

    yield   # ----------------- 여기까지가 startup -----------------

    # 🔥 shutdown 영역 (필요하면 사용)
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)     # server가 만들어짐

@app.get('/')
def root():
    return {
        "message": "Hello!"
    }

@app.post("/infer")
# async는 비동기 처리
async def infer(file:UploadFile = File(...)):                   # body: form-data
    
    allowed_exts = ["jpg", "jpeg", "png", "bmp", "webp"]
    
    
    if not file.filename:
        return {"error": "파일명이 없습니다."}
    
    ext = file.filename.split(".")[-1].lower()
    
    if ext not in allowed_exts:
        return {"error" : "이미지 파일을 업로드하세요!"}
    
    
    # 무조건 데이터는 저장해야 한다. MLops관점에서: 평가 및 재학습
    # 이미지를 uuid로 고유한 이름으로 변경한 뒤 서버가 아닌 클라우드에 저장해야함
    newfile_name = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join("upload_img", newfile_name)
    
    img = await file.read()
    
    with open(file_path, "wb") as buffer:
        buffer.write(img)           # file.file은 파일obj, buffer는 새로운 파일obj
        
    #------------------- 추론 코드 ------------------------
    
    #img = await file.read()    # 위에서 이미 파일 버퍼를 읽어왔기 때문에 비어있어서 에러남.
    pil_img = Image.open(io.BytesIO(img)).convert("RGB")
    #pil_img = Image.open(file_path).convert("RGB")
    
    input_tensor = transform_test(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = app.state.model(input_tensor)
        result = torch.argmax(pred, dim=1).item()
    
    model_class = ['해린','장원영','카리나']
    
    return {
        "result": model_class[result],
        "index": result,
        "filename": newfile_name
    }
    
    