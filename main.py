from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from datetime import datetime
import os
import shutil

# Ensure the working directory is set to the backend folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models import Base, TrafficRecord
from detector import process_media, determine_traffic_status

# DB Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./traffic.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)

def get_db():
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Vehicle Counting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/upload")
async def upload_media(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    vehicle_count, out_path = process_media(file_location, is_video=is_video)
    status = determine_traffic_status(vehicle_count)
    
    static_url = "/" + out_path.replace("\\", "/") if out_path else ""
    
    record = TrafficRecord(
        filename=file.filename,
        timestamp=datetime.utcnow(),
        vehicle_count=vehicle_count,
        traffic_status=status
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    return {
        "id": record.id,
        "filename": record.filename,
        "timestamp": record.timestamp,
        "vehicle_count": record.vehicle_count,
        "traffic_status": record.traffic_status,
        "output_url": static_url
    }

@app.get("/api/dashboard-data")
def get_dashboard_data(db: Session = Depends(get_db)):
    records = db.query(TrafficRecord).order_by(TrafficRecord.timestamp.desc()).all()
    total_vehicles = sum(r.vehicle_count for r in records)
    total_clear = sum(1 for r in records if r.traffic_status == "Clear")
    total_heavy = sum(1 for r in records if r.traffic_status == "Heavy")
    
    return {
        "records": records,
        "stats": {
            "total_processed": len(records),
            "total_vehicles_detected": total_vehicles,
            "clear_count": total_clear,
            "heavy_count": total_heavy
        }
    }

# Serve the frontend
@app.get("/")
def read_root():
    return FileResponse("../frontend/index.html")

# Mount frontend directory for everything else
app.mount("/assets", StaticFiles(directory="../frontend"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

