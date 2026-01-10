import shutil
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ad_engine import app as ad_workflow 

app = FastAPI()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set in environment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force creation of tmp dir (just in case)
os.makedirs("/tmp", exist_ok=True)

@app.get("/")
def home():
    return {"message": "✅ AdForge Backend is Running!", "status": "Online"}

# -----------------------------
@app.post("/generate")
@app.post("/generate-ad")

async def generate_ad(
    request: Request,
    image: UploadFile = File(...), 
    prompt: str = Form(...)
):
    # Auth
    client_key = request.headers.get("x-api-key")
    if client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # 1. Save Upload to /tmp
    temp_filename = f"/tmp/{image.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # 2. Run Graph
        inputs = {
            "product_image_path": temp_filename,
            "user_request": prompt,
            "target_width" : 1024,
            "target_height" : 1024
        }
        
        result = ad_workflow.invoke(inputs)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
            
        # 3. Get Final Image
        # Logic: If final exists, use it. Else use background.
        final = result.get("best_image_url")
        if not final:
            final = result.get("image_url_for_AD")
            
        return {
            "status": "success",
            "image_url": final,
            "headline": result.get("headline", "Ad Generated")
        }

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=7860)