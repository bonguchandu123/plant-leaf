from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File,  WebSocket, WebSocketDisconnect, Query,Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import cloudinary
import cloudinary.uploader
from PIL import Image
import numpy as np
import io
import google.generativeai as genai
from config import settings
from models import *
import tensorflow as tf
from bson import ObjectId
import tempfile
import requests
import uuid
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict,Any
from datetime import datetime
# Add these new imports to your existing imports section

import base64
import stripe
import os
from telugu_translations import TeluguTranslations, get_translator, get_user_language

# Initialize translator
translator = TeluguTranslations()



# After database initialization (after `db = client[settings.DATABASE_NAME]`)




# from fastapi import FastAPI


# Initialize FastAPI
app = FastAPI(title="Organic Advisory System API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MongoDB Connection
client = AsyncIOMotorClient(settings.MONGODB_URL)
db = client[settings.DATABASE_NAME]

# Cloudinary Configuration
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)

# Gemini AI Configuration
genai.configure(api_key=settings.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-flash-latest')

# Load AI Model for Disease Detection
disease_model = tf.keras.models.load_model(settings.MODEL_PATH)

# Load class names
with open(settings.CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]


security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()


# ============= Authentication Utilities =============
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Convert ObjectId to string
        user["_id"] = str(user["_id"])
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate user"
        )

def require_role(required_role: str):
    """Dependency to check user role"""
    async def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user.get("role") != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# ============= AI Utilities =============
async def predict_disease(image_file: UploadFile):
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    predictions = disease_model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return predicted_class, confidence

async def generate_organic_solution(disease_name: str, crop_name: str):
    prompt = f"""
    Generate a detailed organic treatment solution for {disease_name} affecting {crop_name}.
    
    Please provide:
    1. Disease overview in simple language
    2. Step-by-step organic treatment using locally available materials in India
    3. Preventive measures
    4. Estimated time for recovery
    5. Materials needed with local names (Telugu/Hindi)
    
    Focus on sustainable, eco-friendly, and traditional organic methods.
    """
    
    response = gemini_model.generate_content(prompt)
    return response.text
# Replace the signup endpoint in your main.py


# Add language preference endpoint
@app.put("/api/users/language")
async def update_language_preference(
    language_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update user's language preference"""
    language = language_data.get("language", "telugu")
    
    # Normalize language
    if language in ["te", "తెలుగు"]:
        language = "telugu"
    elif language in ["en"]:
        language = "english"
    
    if language not in ["telugu", "english", "hindi"]:
        raise HTTPException(status_code=400, detail="Invalid language")
    
    await db.users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"language_preference": language}}
    )
    
    return {
        "message": "Language updated" if language == "english" else "భాష నవీకరించబడింది",
        "language": language
    }


@app.post("/auth/signup")
async def signup(user_data: dict):
    """User signup with proper validation and location data"""
    
    # Extract data
    name = user_data.get("name")
    phone = user_data.get("phone")
    email = user_data.get("email")
    password = user_data.get("password")
    role = user_data.get("role", "farmer")
    
    # Validation
    if not all([name, phone, password]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Name, phone, and password are required"
        )
    
    # Check if user exists
    existing_user = await db.users.find_one({
        "$or": [
            {"phone": phone},
            {"email": email} if email else {}
        ]
    })
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this phone or email already exists"
        )
    
    # Get location data with defaults
    village = user_data.get("village", "")
    district = user_data.get("district", "Visakhapatnam")
    state = user_data.get("state", "Andhra Pradesh")
    
    # Validate district is not a placeholder
    if district.lower() in ['districtname', 'district', '']:
        district = "Visakhapatnam"
    
    # Create user document
    user_doc = {
        "name": name,
        "phone": phone,
        "email": email,
        "password_hash": get_password_hash(password),
        "role": role,
        "language_preference": user_data.get("language_preference", "telugu"),
        "village": village,
        "district": district,
        "state": state,
        "badges": [],
        "streak_count": 0,
        "last_active": datetime.utcnow(),
        "created_at": datetime.utcnow()
    }
    
    # Insert user
    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    
    # Create user progress record
    await db.user_progress.insert_one({
        "user_id": user_id,
        "crops_monitored": 0,
        "treatments_applied": 0,
        "success_rate": 0.0,
        "learning_sessions_completed": 0,
        "created_at": datetime.utcnow()
    })
    
    # Create default weather preferences
    await db.weather_preferences.insert_one({
        "user_id": user_id,
        "enable_weather_alerts": True,
        "enable_rainfall_alerts": True,
        "enable_temperature_alerts": True,
        "enable_storm_alerts": True,
        "high_temp_threshold": 35.0,
        "low_temp_threshold": 15.0,
        "heavy_rain_threshold": 50.0,
        "morning_alert": True,
        "evening_alert": True,
        "primary_location": district,
        "additional_locations": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # NEW: Auto-create specialist profile if role is specialist
    if role == "specialist":
        await db.specialist_profiles.insert_one({
            "user_id": user_id,
            "specialization": user_data.get("specialization", ["General Agriculture"]),
            "experience_years": user_data.get("experience_years", 0),
            "qualification": user_data.get("qualification", "Agricultural Expert"),
            "languages": user_data.get("languages", ["telugu", "english"]),
            "crops_expertise": user_data.get("crops_expertise", []),
            "diseases_expertise": user_data.get("diseases_expertise", []),
            "bio": user_data.get("bio", "Agricultural specialist ready to help farmers"),
            "consultation_fee": user_data.get("consultation_fee", 0.0),
            "is_online": False,  # Default to offline, specialist can change later
            "last_active": datetime.utcnow(),
            "average_rating": 0.0,
            "total_consultations": 0,
            "total_ratings": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id}, 
        expires_delta=access_token_expires
    )
    
    # Get created user for response
    created_user = await db.users.find_one({"_id": result.inserted_id})
    
    # Format user response
    user_response = {
        "id": user_id,
        "name": created_user["name"],
        "phone": created_user["phone"],
        "email": created_user.get("email"),
        "role": created_user["role"],
        "language_preference": created_user.get("language_preference", "telugu"),
        "village": created_user.get("village"),
        "district": created_user.get("district"),
        "state": created_user.get("state"),
        "badges": created_user.get("badges", []),
        "streak_count": created_user.get("streak_count", 0),
        "created_at": created_user["created_at"].isoformat()
    }
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }
@app.post("/auth/login")
async def login(login_data: dict):
    """User login with phone/email and password"""
    
    # Extract credentials
    identifier = login_data.get("phone") or login_data.get("email")
    password = login_data.get("password")
    
    if not identifier or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Phone/email and password are required"
        )
    
    # Find user by phone or email
    user = await db.users.find_one({
        "$or": [
            {"phone": identifier},
            {"email": identifier}
        ]
    })
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Update last active
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_active": datetime.utcnow()}}
    )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["_id"])},
        expires_delta=access_token_expires
    )
    
    # Format user response
    user_response = {
        "id": str(user["_id"]),
        "name": user["name"],
        "phone": user["phone"],
        "email": user.get("email"),
        "role": user["role"],
        "language_preference": user.get("language_preference", "telugu"),
        "village": user.get("village"),
        "district": user.get("district"),
        "badges": user.get("badges", []),
        "streak_count": user.get("streak_count", 0),
        "last_active": user["last_active"].isoformat() if user.get("last_active") else None
    }
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.post("/auth/reset-password")
async def reset_password(reset_data: dict):
    """Password reset request"""
    
    identifier = reset_data.get("phone") or reset_data.get("email")
    
    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Phone or email is required"
        )
    
    user = await db.users.find_one({
        "$or": [
            {"phone": identifier},
            {"email": identifier}
        ]
    })
    
    if not user:
        # Don't reveal if user exists
        return {
            "message": "If a user with this phone/email exists, reset instructions have been sent"
        }
    
    # In production: Send OTP via SMS or email
    # For now, just acknowledge
    return {
        "message": "If a user with this phone/email exists, reset instructions have been sent"
    }


@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information"""
    
    # Get user progress
    progress = await db.user_progress.find_one({"user_id": current_user["_id"]})
    
    return {
        "user": {
            "id": current_user["_id"],
            "name": current_user["name"],
            "phone": current_user["phone"],
            "email": current_user.get("email"),
            "role": current_user["role"],
            "language_preference": current_user.get("language_preference", "telugu"),
            "village": current_user.get("village"),
            "district": current_user.get("district"),
            "badges": current_user.get("badges", []),
            "streak_count": current_user.get("streak_count", 0)
        },
        "progress": {
            "crops_monitored": progress["crops_monitored"] if progress else 0,
            "treatments_applied": progress["treatments_applied"] if progress else 0,
            "success_rate": progress["success_rate"] if progress else 0,
            "learning_sessions_completed": progress["learning_sessions_completed"] if progress else 0
        } if progress else None
    }


@app.get("/api/dashboard/farmer")
async def get_farmer_dashboard(current_user: dict = Depends(get_current_user)):
    """Farmer dashboard with Telugu support"""
    
    if current_user["role"] != "farmer":
        raise HTTPException(
            status_code=403,
            detail="Access forbidden - farmer role required"
        )
    
    user_id = current_user["_id"]
    user_language = current_user.get("language_preference", "telugu")
    
    # Get stats
    crop_photos = await db.crop_photos.count_documents({"user_id": user_id})
    treatments = await db.treatment_submissions.count_documents({"user_id": user_id})
    progress = await db.user_progress.find_one({"user_id": user_id})
    
    # Get recent data
    recent_photos = await db.crop_photos.find(
        {"user_id": user_id}
    ).sort("uploaded_at", -1).limit(5).to_list(5)
    
    recent_alerts = await db.weather_alerts.find(
        {"location": current_user.get("district", "")}
    ).sort("timestamp", -1).limit(5).to_list(5)
    
    response_data = {
        "user": {
            "name": current_user["name"],
            "village": current_user.get("village"),
            "district": current_user.get("district"),
            "badges": current_user.get("badges", []),
            "streak": current_user.get("streak_count", 0),
            "language_preference": user_language
        },
        "stats": {
            "crops_monitored": progress["crops_monitored"] if progress else 0,
            "treatments_applied": progress["treatments_applied"] if progress else 0,
            "success_rate": progress["success_rate"] if progress else 0,
            "total_photos": crop_photos
        },
        "recent_photos": [
            {
                "id": str(p["_id"]),
                "image_url": p["image_url"],
                "disease": p.get("disease"),
                "confidence": p.get("confidence_score"),
                "uploaded_at": p["uploaded_at"].isoformat()
            } for p in recent_photos
        ],
        "weather_alerts": [
            {
                "id": str(a["_id"]),
                "type": a["alert_type"],
                "message": a["message"],
                "recommended_action": a.get("recommended_action"),
                "timestamp": a["timestamp"].isoformat()
            } for a in recent_alerts
        ]
    }
    
    # Add Telugu labels if user prefers Telugu
    if user_language == "telugu":
        response_data["_labels"] = {
            "crops_monitored": "పర్యవేక్షించిన పంటలు",
            "treatments_applied": "అన్వయించిన చికిత్సలు",
            "success_rate": "విజయ రేటు",
            "total_photos": "మొత్తం ఫోటోలు",
            "recent_photos": "ఇటీవలి ఫోటోలు",
            "weather_alerts": "వాతావరణ హెచ్చరికలు",
            "disease": "వ్యాధి",
            "confidence": "విశ్వాసం",
            "name": "పేరు",
            "village": "గ్రామం",
            "district": "జిల్లా",
            "badges": "బ్యాడ్జ్‌లు",
            "streak": "వరుస రోజులు"
        }
    
    return response_data

@app.get("/api/dashboard/specialist")
async def get_specialist_dashboard(
    current_user: dict = Depends(require_role("specialist"))
):
    """Specialist dashboard with proper authentication"""
    
    # Get statistics
    total_farmers = await db.users.count_documents({"role": "farmer"})
    total_diagnoses = await db.ai_interactions.count_documents({})
    total_treatments = await db.treatments.count_documents({})
    
    # Get recent interactions
    recent_interactions = await db.ai_interactions.find().sort(
        "timestamp", -1
    ).limit(10).to_list(10)
    
    # Get disease statistics
    disease_stats = await db.ai_interactions.aggregate([
        {"$group": {"_id": "$disease_prediction", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]).to_list(5)
    
    # Get active users (active in last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    active_users = await db.users.count_documents({
        "last_active": {"$gte": week_ago}
    })
    
    return {
        "stats": {
            "total_farmers": total_farmers,
            "total_diagnoses": total_diagnoses,
            "total_treatments": total_treatments,
            "active_users": active_users
        },
        "recent_interactions": [
            {
                "id": str(i["_id"]),
                "disease": i["disease_prediction"],
                "confidence": i["confidence_score"],
                "timestamp": i["timestamp"].isoformat()
            } for i in recent_interactions
        ],
        "disease_trends": [
            {
                "disease": d["_id"],
                "count": d["count"]
            } for d in disease_stats
        ]
    }


@app.put("/api/users/language")
async def update_language_preference(
    language_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update user's language preference"""
    language = language_data.get("language", "telugu")
    
    # Normalize language
    if language in ["te", "తెలుగు"]:
        language = "telugu"
    elif language in ["en"]:
        language = "english"
    
    if language not in ["telugu", "english", "hindi"]:
        raise HTTPException(status_code=400, detail="Invalid language")
    
    await db.users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"language_preference": language}}
    )
    
    return {
        "message": "Language updated" if language == "english" else "భాష నవీకరించబడింది",
        "language": language
    }


# Modify the crop photo upload endpoint to return localized responses
@app.post("/api/crop-photos/upload")
async def upload_crop_photo(
    file: UploadFile = File(...),
    crop_id: Optional[str] = None,
    location: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Upload and analyze crop photo with language support"""
    
    # Get user's language preference
    user_language = current_user.get("language_preference", "telugu")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        error_msg = "దయచేసి చెల్లుబాటు అయ్యే చిత్ర ఫైల్‌ను ఎంచుకోండి (JPEG, PNG, JPG మాత్రమే)" if user_language == "telugu" else "Invalid file type. Only JPEG, PNG, and JPG images are allowed"
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        error_msg = "ఫైల్ పరిమాణం చాలా పెద్దది. గరిష్ట పరిమాణం 10MB" if user_language == "telugu" else "File size too large. Maximum size is 10MB"
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Reset file pointer for disease prediction
    await file.seek(0)
    
    try:
        # Predict disease using AI model
        disease_name, confidence = await predict_disease(file)
        print(f"Disease detected: {disease_name} with confidence: {confidence}")
        
    except Exception as e:
        print(f"Error predicting disease: {e}")
        error_msg = "చిత్రాన్ని విశ్లేషించడంలో విఫలమైంది. దయచేసి మళ్ళీ ప్రయత్నించండి." if user_language == "telugu" else "Failed to analyze image. Please try again."
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Upload to Cloudinary
    try:
        extension_map = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg", 
            "image/png": "png"
        }
        file_format = extension_map.get(file.content_type, "jpg")
        
        upload_result = cloudinary.uploader.upload(
            contents,
            folder="crop_photos",
            resource_type="image",
            format=file_format,
            allowed_formats=['jpg', 'png', 'jpeg'],
            transformation=[
                {'width': 1200, 'height': 1200, 'crop': 'limit'},
                {'quality': 'auto'}
            ]
        )
        
        image_url = upload_result.get("secure_url")
        
        if not image_url or image_url.lower().endswith('.pdf'):
            raise Exception("Invalid upload result")
            
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        error_msg = "చిత్రాన్ని అప్‌లోడ్ చేయడంలో విఫలమైంది" if user_language == "telugu" else f"Failed to upload image: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Get crop name
    crop_name = "Unknown"
    if crop_id:
        try:
            crop = await db.crops.find_one({"_id": ObjectId(crop_id)})
            if crop:
                crop_name = crop["name"]
        except Exception as e:
            print(f"Error fetching crop: {e}")
    
    # Generate organic solution (use Telugu prompt if language is Telugu)
    try:
        if user_language == "telugu":
            prompt = f"""
            {crop_name} పంటను ప్రభావితం చేసే {disease_name} కోసం వివరణాత్మక సేంద్రీయ చికిత్స పరిష్కారాన్ని తెలుగులో రూపొందించండి.
            
            దయచేసి అందించండి:
            1. సాధారణ భాషలో వ్యాధి యొక్క సంక్షిప్త వివరణ
            2. భారతదేశంలో స్థానికంగా అందుబాటులో ఉన్న పదార్థాలను ఉపయోగించి దశల వారీగా సేంద్రీయ చికిత్స
            3. నివారణ చర్యలు
            4. కోలుకోవడానికి అంచనా సమయం
            5. స్థానిక పేర్లతో (తెలుగు/హిందీ) అవసరమైన పదార్థాలు
            
            స్థిరమైన, పర్యావరణ అనుకూల మరియు సాంప్రదాయ సేంద్రీయ పద్ధతులపై దృష్టి పెట్టండి.
            """
        else:
            prompt = f"""
            Generate a detailed organic treatment solution for {disease_name} affecting {crop_name}.
            
            Please provide:
            1. Disease overview in simple language
            2. Step-by-step organic treatment using locally available materials in India
            3. Preventive measures
            4. Estimated time for recovery
            5. Materials needed with local names (Telugu/Hindi)
            
            Focus on sustainable, eco-friendly, and traditional organic methods.
            """
        
        response = gemini_model.generate_content(prompt)
        organic_solution = response.text
        
    except Exception as e:
        print(f"Error generating solution: {e}")
        organic_solution = "చికిత్స సిఫార్సును ఉత్పత్తి చేయడం సాధ్యం కాలేదు" if user_language == "telugu" else "Unable to generate treatment recommendation at this time."
    
    # Prepare photo data
    photo_data = {
        "user_id": str(current_user["_id"]),
        "crop_id": crop_id,
        "image_url": image_url,
        "location": location or current_user.get("district", ""),
        "uploaded_at": datetime.utcnow(),
        "disease": disease_name,
        "confidence_score": confidence,
        "suggested_treatment": organic_solution,
        "severity": "medium",
        "status": "active"
    }
    
    # Insert photo record
    try:
        result = await db.crop_photos.insert_one(photo_data)
        photo_id = str(result.inserted_id)
    except Exception as e:
        print(f"Error saving photo: {e}")
        error_msg = "విశ్లేషణ డేటాను సేవ్ చేయడంలో విఫలమైంది" if user_language == "telugu" else "Failed to save analysis data"
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Record AI interaction
    await db.ai_interactions.insert_one({
        "user_id": str(current_user["_id"]),
        "image_id": photo_id,
        "disease_prediction": disease_name,
        "confidence_score": confidence,
        "suggested_treatment": organic_solution,
        "timestamp": datetime.utcnow()
    })
    
    # Update user progress
    await db.user_progress.update_one(
        {"user_id": str(current_user["_id"])},
        {
            "$inc": {"crops_monitored": 1},
            "$set": {"last_updated": datetime.utcnow()}
        },
        upsert=True
    )
    
    # Broadcast to WebSocket
    try:
        await manager.broadcast({
            "type": "photo_uploaded",
            "user_id": str(current_user["_id"]),
            "user_name": current_user.get("name", "Unknown"),
            "disease": disease_name,
            "confidence": confidence,
            "photo_id": photo_id
        })
    except Exception as e:
        print(f"Error broadcasting: {e}")
    
    # Return localized response
    success_msg = "ఫోటో విజయవంతంగా విశ్లేషించబడింది" if user_language == "telugu" else "Photo analyzed successfully"
    
    return {
        "photo_id": photo_id,
        "image_url": image_url,
        "diagnosis": {
            "disease": disease_name,
            "confidence": round(confidence, 3),
            "treatment": organic_solution
        },
        "message": success_msg,
        "_labels": {
            "disease": translator.get("disease") if user_language == "telugu" else "Disease",
            "confidence": translator.get("confidence") if user_language == "telugu" else "Confidence",
            "treatment": translator.get("treatment") if user_language == "telugu" else "Treatment"
        } if user_language == "telugu" else None
    }


app.get("/api/crop-photos/search")
async def search_crop_analyses(
    query: Optional[str] = None,
    disease: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    location: Optional[str] = None,
    crop_id: Optional[str] = None,
    sort_by: str = "uploaded_at",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 20,
    current_user: dict = Depends(get_current_user)
):
    search_query = {"user_id": str(current_user["_id"])}
    
    if disease:
        search_query["disease"] = {"$regex": disease, "$options": "i"}
    
    if min_confidence is not None:
        search_query["confidence_score"] = {"$gte": min_confidence}
    if max_confidence is not None:
        if "confidence_score" not in search_query:
            search_query["confidence_score"] = {}
        search_query["confidence_score"]["$lte"] = max_confidence
    
    if severity:
        search_query["severity"] = severity
    
    if status:
        search_query["status"] = status
    
    if start_date:
        search_query["uploaded_at"] = {"$gte": datetime.fromisoformat(start_date)}
    if end_date:
        if "uploaded_at" not in search_query:
            search_query["uploaded_at"] = {}
        search_query["uploaded_at"]["$lte"] = datetime.fromisoformat(end_date)
    
    if location:
        search_query["location"] = {"$regex": location, "$options": "i"}
    
    if crop_id:
        search_query["crop_id"] = crop_id
    
    if query:
        search_query["$or"] = [
            {"disease": {"$regex": query, "$options": "i"}},
            {"suggested_treatment": {"$regex": query, "$options": "i"}},
            {"location": {"$regex": query, "$options": "i"}}
        ]
    
    total = await db.crop_photos.count_documents(search_query)
    
    sort_direction = -1 if sort_order == "desc" else 1
    skip = (page - 1) * page_size
    
    photos = await db.crop_photos.find(search_query).sort(
        sort_by, sort_direction
    ).skip(skip).limit(page_size).to_list(page_size)
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "results": [
            {
                "id": str(p["_id"]),
                "image_url": p["image_url"],
                "disease": p.get("disease"),
                "confidence": p.get("confidence_score"),
                "severity": p.get("severity"),
                "status": p.get("status"),
                "location": p.get("location"),
                "uploaded_at": p["uploaded_at"].isoformat()
            } for p in photos
        ]
    }

@app.get("/api/crop-photos/history/{photo_id}")
async def get_photo_history(
    photo_id: str,
    current_user: dict = Depends(get_current_user)
):
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    if photo["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get all AI interactions for this photo
    interactions = await db.ai_interactions.find(
        {"image_id": photo_id}
    ).sort("timestamp", -1).to_list(100)
    
    # Get feedback
    feedback = await db.analysis_feedback.find(
        {"photo_id": photo_id}
    ).sort("submitted_at", -1).to_list(100)
    
    # Get related treatments
    treatments = await db.treatment_submissions.find(
        {"photo_id": photo_id}
    ).sort("submitted_at", -1).to_list(100)
    
    return {
        "photo": {
            "id": str(photo["_id"]),
            "image_url": photo["image_url"],
            "current_disease": photo.get("disease"),
            "current_confidence": photo.get("confidence_score"),
            "uploaded_at": photo["uploaded_at"].isoformat()
        },
        "ai_interactions": [
            {
                "disease": i["disease_prediction"],
                "confidence": i["confidence_score"],
                "treatment": i.get("suggested_treatment"),
                "timestamp": i["timestamp"].isoformat()
            } for i in interactions
        ],
        "feedback": [
            {
                "rating": f.get("rating"),
                "is_accurate": f.get("is_accurate"),
                "comments": f.get("comments"),
                "submitted_at": f["submitted_at"].isoformat()
            } for f in feedback
        ],
        "treatments": [
            {
                "treatment_id": t["treatment_id"],
                "status": t["status"],
                "outcome": t.get("outcome"),
                "submitted_at": t["submitted_at"].isoformat()
            } for t in treatments
        ]
    }

@app.get("/api/crop-photos/similar/{photo_id}")
async def find_similar_analyses(
    photo_id: str,
    limit: int = 5,
    current_user: dict = Depends(get_current_user)
):
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    disease = photo.get("disease")
    confidence = photo.get("confidence_score", 0)
    
    similar_query = {
        "_id": {"$ne": ObjectId(photo_id)},
        "disease": disease,
        "confidence_score": {
            "$gte": confidence - 0.2,
            "$lte": confidence + 0.2
        }
    }
    
    similar_photos = await db.crop_photos.find(similar_query).limit(limit).to_list(limit)
    
    return [
        {
            "id": str(p["_id"]),
            "image_url": p["image_url"],
            "disease": p.get("disease"),
            "confidence": p.get("confidence_score"),
            "similarity_score": 1 - abs(confidence - p.get("confidence_score", 0)),
            "uploaded_at": p["uploaded_at"].isoformat()
        } for p in similar_photos
    ]


# ============= Treatment Routes =============# ============= Crop Photo Routes =============
@app.post("/api/crop-photos/upload")
async def upload_crop_photo(
    file: UploadFile = File(...),
    crop_id: Optional[str] = None,
    location: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    disease_name, confidence = await predict_disease(file)
    
    await file.seek(0)
    contents = await file.read()
    upload_result = cloudinary.uploader.upload(contents, folder="crop_photos")
    
    crop_name = "Unknown"
    if crop_id:
        crop = await db.crops.find_one({"_id": ObjectId(crop_id)})
        if crop:
            crop_name = crop["name"]
    
    organic_solution = await generate_organic_solution(disease_name, crop_name)
    
    photo_data = {
        "user_id": str(current_user["_id"]),
        "crop_id": crop_id,
        "image_url": upload_result["secure_url"],
        "location": location,
        "uploaded_at": datetime.utcnow(),
        "disease": disease_name,
        "confidence_score": confidence,
        "suggested_treatment": organic_solution,
        "severity": "medium",
        "status": "active"
    }
    
    result = await db.crop_photos.insert_one(photo_data)
    
    await db.ai_interactions.insert_one({
        "user_id": str(current_user["_id"]),
        "image_id": str(result.inserted_id),
        "disease_prediction": disease_name,
        "confidence_score": confidence,
        "suggested_treatment": organic_solution,
        "timestamp": datetime.utcnow()
    })
    
    await db.user_progress.update_one(
        {"user_id": str(current_user["_id"])},
        {"$inc": {"crops_monitored": 1}}
    )
    
    await manager.broadcast({
        "type": "photo_uploaded",
        "user_id": str(current_user["_id"]),
        "disease": disease_name,
        "confidence": confidence
    })
    
    return {
        "photo_id": str(result.inserted_id),
        "image_url": upload_result["secure_url"],
        "diagnosis": {
            "disease": disease_name,
            "confidence": confidence,
            "treatment": organic_solution
        }
    }

@app.get("/api/crop-photos")
async def get_crop_photos(
    status: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    query = {"user_id": str(current_user["_id"])}
    if status:
        query["status"] = status
    
    photos = await db.crop_photos.find(query).sort("uploaded_at", -1).limit(limit).to_list(limit)
    return [
        {
            "id": str(p["_id"]),
            "image_url": p["image_url"],
            "disease": p.get("disease"),
            "confidence": p.get("confidence_score"),
            "severity": p.get("severity", "medium"),
            "status": p.get("status", "active"),
            "uploaded_at": p["uploaded_at"].isoformat()
        } for p in photos
    ]

@app.get("/api/crop-photos/{photo_id}")
async def get_crop_photo_detail(photo_id: str, current_user: dict = Depends(get_current_user)):
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    return {
        "id": str(photo["_id"]),
        "image_url": photo["image_url"],
        "disease": photo.get("disease"),
        "confidence": photo.get("confidence_score"),
        "treatment": photo.get("suggested_treatment"),
        "severity": photo.get("severity", "medium"),
        "status": photo.get("status", "active"),
        "location": photo.get("location"),
        "crop_id": photo.get("crop_id"),
        "uploaded_at": photo["uploaded_at"].isoformat()
    }

@app.put("/api/crop-photos/{photo_id}")
async def update_crop_photo(
    photo_id: str,
    update_data: dict,
    current_user: dict = Depends(get_current_user)
):
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    if photo["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to update this photo")
    
    allowed_fields = ["status", "severity", "notes"]
    update_fields = {k: v for k, v in update_data.items() if k in allowed_fields}
    update_fields["updated_at"] = datetime.utcnow()
    
    await db.crop_photos.update_one(
        {"_id": ObjectId(photo_id)},
        {"$set": update_fields}
    )
    
    return {"message": "Photo updated successfully"}

@app.delete("/api/crop-photos/{photo_id}")
async def delete_crop_photo(
    photo_id: str,
    current_user: dict = Depends(get_current_user)
):
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    if photo["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to delete this photo")
    
    await db.crop_photos.delete_one({"_id": ObjectId(photo_id)})
    await db.ai_interactions.delete_many({"image_id": photo_id})
    
    return {"message": "Photo deleted successfully"}

@app.post("/api/crop-photos/{photo_id}/reanalyze")
async def reanalyze_crop_photo(
    photo_id: str,
    current_user: dict = Depends(get_current_user)
):
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    if photo["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Download image from Cloudinary
    import requests
    response = requests.get(photo["image_url"])
    image_bytes = io.BytesIO(response.content)
    
    # Create temporary UploadFile
    temp_file = UploadFile(filename="temp.jpg", file=image_bytes)
    
    disease_name, confidence = await predict_disease(temp_file)
    
    crop_name = "Unknown"
    if photo.get("crop_id"):
        crop = await db.crops.find_one({"_id": ObjectId(photo["crop_id"])})
        if crop:
            crop_name = crop["name"]
    
    organic_solution = await generate_organic_solution(disease_name, crop_name)
    
    await db.crop_photos.update_one(
        {"_id": ObjectId(photo_id)},
        {"$set": {
            "disease": disease_name,
            "confidence_score": confidence,
            "suggested_treatment": organic_solution,
            "updated_at": datetime.utcnow()
        }}
    )
    
    await db.ai_interactions.insert_one({
        "user_id": str(current_user["_id"]),
        "image_id": photo_id,
        "disease_prediction": disease_name,
        "confidence_score": confidence,
        "suggested_treatment": organic_solution,
        "timestamp": datetime.utcnow()
    })
    
    return {
        "diagnosis": {
            "disease": disease_name,
            "confidence": confidence,
            "treatment": organic_solution
        }
    }

@app.get("/api/crop-photos/statistics/summary")
async def get_analysis_statistics(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    
    total_analyses = await db.crop_photos.count_documents({"user_id": user_id})
    
    # Disease distribution
    disease_pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$disease", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    disease_stats = await db.crop_photos.aggregate(disease_pipeline).to_list(5)
    
    # Severity distribution
    severity_pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
    ]
    severity_stats = await db.crop_photos.aggregate(severity_pipeline).to_list(10)
    
    # Average confidence
    confidence_pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence_score"}}}
    ]
    confidence_result = await db.crop_photos.aggregate(confidence_pipeline).to_list(1)
    avg_confidence = confidence_result[0]["avg_confidence"] if confidence_result else 0
    
    # Recent trends (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_analyses = await db.crop_photos.count_documents({
        "user_id": user_id,
        "uploaded_at": {"$gte": thirty_days_ago}
    })
    
    return {
        "total_analyses": total_analyses,
        "recent_analyses_30d": recent_analyses,
        "average_confidence": round(avg_confidence, 2),
        "disease_distribution": [
            {"disease": d["_id"], "count": d["count"]} for d in disease_stats
        ],
        "severity_distribution": [
            {"severity": s["_id"], "count": s["count"]} for s in severity_stats
        ]
    }

@app.post("/api/crop-photos/{photo_id}/feedback")
async def submit_analysis_feedback(
    photo_id: str,
    feedback_data: dict,
    current_user: dict = Depends(get_current_user)
):
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    feedback_doc = {
        "photo_id": photo_id,
        "user_id": str(current_user["_id"]),
        "rating": feedback_data.get("rating"),
        "is_accurate": feedback_data.get("is_accurate"),
        "comments": feedback_data.get("comments"),
        "actual_disease": feedback_data.get("actual_disease"),
        "submitted_at": datetime.utcnow()
    }
    
    result = await db.analysis_feedback.insert_one(feedback_doc)
    
    return {"message": "Feedback submitted successfully", "feedback_id": str(result.inserted_id)}

# ============= Treatment Routes =============
@app.post("/api/treatments")
async def create_treatment(
    crop_id: Optional[str],
    disease_or_pest_id: Optional[str],
    step_by_step_guide: List[str],
    ingredients_local_availability: List[dict],
    estimated_time: str,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "specialist":
        raise HTTPException(status_code=403, detail="Only specialists can create treatments")
    
    treatment_data = {
        "crop_id": crop_id,
        "disease_or_pest_id": disease_or_pest_id,
        "step_by_step_guide": step_by_step_guide,
        "ingredients_local_availability": ingredients_local_availability,
        "estimated_time": estimated_time,
        "seasonal_relevance": [],
        "media_urls": [],
        "created_by": str(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    result = await db.treatments.insert_one(treatment_data)
    return {"treatment_id": str(result.inserted_id), "message": "Treatment created successfully"}

@app.get("/api/treatments")
async def get_treatments(crop_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    query = {}
    if crop_id:
        query["crop_id"] = crop_id
    
    treatments = await db.treatments.find(query).to_list(100)
    return [
        {
            "id": str(t["_id"]),
            "crop_id": t.get("crop_id"),
            "steps": t["step_by_step_guide"],
            "estimated_time": t["estimated_time"],
            "created_at": t["created_at"].isoformat()
        } for t in treatments
    ]

@app.post("/api/treatments/apply")
async def apply_treatment(
    treatment_id: str,
    notes: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    submission_data = {
        "user_id": str(current_user["_id"]),
        "treatment_id": treatment_id,
        "status": "applied",
        "outcome": None,
        "notes": notes,
        "photo_before_after": [],
        "submitted_at": datetime.utcnow()
    }
    
    result = await db.treatment_submissions.insert_one(submission_data)
    
    await db.user_progress.update_one(
        {"user_id": str(current_user["_id"])},
        {"$inc": {"treatments_applied": 1}}
    )
    
    await manager.broadcast({
        "type": "treatment_applied",
        "user_id": str(current_user["_id"]),
        "treatment_id": treatment_id
    })
    
    return {"submission_id": str(result.inserted_id), "message": "Treatment applied successfully"}


# ============= Community Routes =============
# Add these routes to your main.py file

# ============= Enhanced Community Routes with Crop Analysis Integration =============

# @app.post("/api/community/posts")
# async def create_post(
#     post_data: dict,
#     current_user: dict = Depends(get_current_user)
# ):
#     """Create a community post (can include crop analysis reference)"""
    
#     title = post_data.get("title")
#     content_text = post_data.get("content_text")
#     tags = post_data.get("tags", [])
#     photo_id = post_data.get("photo_id")  # Reference to crop analysis
#     media_urls = post_data.get("media_urls", [])
    
#     if not title or not content_text:
#         raise HTTPException(
#             status_code=400,
#             detail="Title and content are required"
#         )
    
#     # If photo_id is provided, get analysis details
#     analysis_data = None
#     if photo_id:
#         photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
#         if photo:
#             analysis_data = {
#                 "photo_id": photo_id,
#                 "image_url": photo["image_url"],
#                 "disease": photo.get("disease"),
#                 "confidence": photo.get("confidence_score")
#             }
#             # Auto-add image URL to media
#             if photo["image_url"] not in media_urls:
#                 media_urls.append(photo["image_url"])
    
#     post_doc = {
#         "title": title,
#         "content_text": content_text,
#         "media_urls": media_urls,
#         "author_id": str(current_user["_id"]),
#         "author_name": current_user["name"],
#         "location": current_user.get("village"),
#         "tags": tags,
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow(),
#         "comments": [],
#         "likes": 0,
#         "views": 0,
#         "helpful_count": 0,
#         # Analysis reference
#         "analysis_reference": analysis_data,
#         "is_question": post_data.get("is_question", False),
#         "is_solved": False
#     }
    
#     result = await db.community_posts.insert_one(post_doc)
    
#     # Broadcast to WebSocket
#     await manager.broadcast({
#         "type": "new_post",
#         "post": {
#             "id": str(result.inserted_id),
#             "title": title,
#             "author": current_user["name"],
#             "created_at": datetime.utcnow().isoformat()
#         }
#     })
    
#     return {
#         "post_id": str(result.inserted_id),
#         "message": "Post created successfully"
#     }

@app.post("/api/community/posts")
async def create_post(
    post_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create a community post (can include crop analysis reference or uploaded images)"""
    
    title = post_data.get("title")
    content_text = post_data.get("content_text")
    tags = post_data.get("tags", [])
    photo_id = post_data.get("photo_id")  # Reference to crop analysis
    media_urls = post_data.get("media_urls", [])  # Uploaded image URLs
    is_question = post_data.get("is_question", False)
    
    if not title or not content_text:
        raise HTTPException(
            status_code=400,
            detail="Title and content are required"
        )
    
    # If photo_id is provided, get analysis details
    analysis_data = None
    if photo_id:
        photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
        if photo:
            analysis_data = {
                "photo_id": photo_id,
                "image_url": photo["image_url"],
                "disease": photo.get("disease"),
                "confidence": photo.get("confidence_score")
            }
            # Auto-add image URL to media
            if photo["image_url"] not in media_urls:
                media_urls.insert(0, photo["image_url"])
    
    post_doc = {
        "title": title,
        "content_text": content_text,
        "media_urls": media_urls,
        "author_id": str(current_user["_id"]),
        "author_name": current_user["name"],
        "location": current_user.get("village"),
        "tags": tags,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "comments": [],
        "likes": 0,
        "views": 0,
        "helpful_count": 0,
        "analysis_reference": analysis_data,
        "is_question": is_question,
        "is_solved": False
    }
    
    result = await db.community_posts.insert_one(post_doc)
    
    # Broadcast to WebSocket
    await manager.broadcast({
        "type": "new_post",
        "post": {
            "id": str(result.inserted_id),
            "title": title,
            "author": current_user["name"],
            "created_at": datetime.utcnow().isoformat()
        }
    })
    
    return {
        "post_id": str(result.inserted_id),
        "message": "Post created successfully"
    }


# Add new endpoint for uploading images to community posts
@app.post("/api/community/upload-image")
async def upload_community_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload image for community post"""
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG, PNG, and JPG images are allowed"
        )
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum size is 10MB"
        )
    
    try:
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            contents,
            folder="community_posts",
            resource_type="image",
            transformation=[
                {'width': 1200, 'height': 1200, 'crop': 'limit'},
                {'quality': 'auto'}
            ]
        )
        
        return {
            "image_url": upload_result.get("secure_url"),
            "message": "Image uploaded successfully"
        }
        
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload image"
        )

# @app.get("/api/community/posts")
# async def get_posts(
#     category: Optional[str] = None,
#     tag: Optional[str] = None,
#     has_analysis: Optional[bool] = None,
#     is_question: Optional[bool] = None,
#     sort_by: str = "created_at",
#     limit: int = 50,
#     current_user: dict = Depends(get_current_user)
# ):
#     """Get community posts with filters"""
#     query = {}
    
#     if category:
#         query["tags"] = {"$in": [category]}
    
#     if tag:
#         query["tags"] = {"$in": [tag]}
    
#     if has_analysis is not None:
#         if has_analysis:
#             query["analysis_reference"] = {"$ne": None}
#         else:
#             query["analysis_reference"] = None
    
#     if is_question is not None:
#         query["is_question"] = is_question
    
#     # Get posts
#     posts = await db.community_posts.find(query).sort(
#         sort_by, -1
#     ).limit(limit).to_list(limit)
    
#     # Get comments count for each post
#     result = []
#     for post in posts:
#         comments_count = await db.community_comments.count_documents(
#             {"post_id": str(post["_id"])}
#         )
        
#         result.append({
#             "id": str(post["_id"]),
#             "title": post["title"],
#             "content": post["content_text"],
#             "author": post.get("author_name", "Unknown"),
#             "author_id": post["author_id"],
#             "location": post.get("location"),
#             "tags": post.get("tags", []),
#             "media_urls": post.get("media_urls", []),
#             "analysis_reference": post.get("analysis_reference"),
#             "is_question": post.get("is_question", False),
#             "is_solved": post.get("is_solved", False),
#             "likes": post.get("likes", 0),
#             "views": post.get("views", 0),
#             "helpful_count": post.get("helpful_count", 0),
#             "comments_count": comments_count,
#             "created_at": post["created_at"].isoformat()
#         })
    
#     return result

@app.get("/api/community/posts")
async def get_posts(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    has_analysis: Optional[bool] = None,
    is_question: Optional[bool] = None,
    sort_by: str = "created_at",
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get community posts with filters"""
    query = {}
    
    if category:
        query["tags"] = {"$in": [category]}
    
    if tag:
        query["tags"] = {"$in": [tag]}
    
    if has_analysis is not None:
        if has_analysis:
            query["analysis_reference"] = {"$ne": None}
        else:
            query["analysis_reference"] = None
    
    if is_question is not None:
        query["is_question"] = is_question
    
    # Get posts
    posts = await db.community_posts.find(query).sort(
        sort_by, -1
    ).limit(limit).to_list(limit)
    
    # Get comments count for each post
    result = []
    for post in posts:
        comments_count = await db.community_comments.count_documents(
            {"post_id": str(post["_id"])}
        )
        
        # Ensure media_urls includes analysis image if present
        media_urls = post.get("media_urls", [])
        if post.get("analysis_reference") and post["analysis_reference"].get("image_url"):
            if post["analysis_reference"]["image_url"] not in media_urls:
                media_urls.insert(0, post["analysis_reference"]["image_url"])
        
        result.append({
            "id": str(post["_id"]),
            "title": post["title"],
            "content": post["content_text"],
            "author": post.get("author_name", "Unknown"),
            "author_id": post["author_id"],
            "location": post.get("location"),
            "tags": post.get("tags", []),
            "media_urls": media_urls,  # Fixed: always includes analysis image
            "analysis_reference": post.get("analysis_reference"),
            "is_question": post.get("is_question", False),
            "is_solved": post.get("is_solved", False),
            "likes": post.get("likes", 0),
            "views": post.get("views", 0),
            "helpful_count": post.get("helpful_count", 0),
            "comments_count": comments_count,
            "created_at": post["created_at"].isoformat()
        })
    
    return result

@app.get("/api/community/posts/{post_id}")
async def get_post_detail(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed post with comments"""
    post = await db.community_posts.find_one({"_id": ObjectId(post_id)})
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Increment view count
    await db.community_posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$inc": {"views": 1}}
    )
    
    # Get comments
    comments = await db.community_comments.find(
        {"post_id": post_id}
    ).sort("created_at", 1).to_list(1000)
    
    # Format comments with user info
    formatted_comments = []
    for comment in comments:
        user = await db.users.find_one({"_id": ObjectId(comment["user_id"])})
        formatted_comments.append({
            "id": str(comment["_id"]),
            "comment_text": comment["comment_text"],
            "user_name": user["name"] if user else "Unknown",
            "user_id": comment["user_id"],
            "media_urls": comment.get("media_urls", []),
            "likes": comment.get("likes", 0),
            "created_at": comment["created_at"].isoformat()
        })
    
    return {
        "id": str(post["_id"]),
        "title": post["title"],
        "content": post["content_text"],
        "author": post.get("author_name", "Unknown"),
        "author_id": post["author_id"],
        "location": post.get("location"),
        "tags": post.get("tags", []),
        "media_urls": post.get("media_urls", []),
        "analysis_reference": post.get("analysis_reference"),
        "is_question": post.get("is_question", False),
        "is_solved": post.get("is_solved", False),
        "likes": post.get("likes", 0),
        "views": post.get("views", 0),
        "helpful_count": post.get("helpful_count", 0),
        "comments": formatted_comments,
        "created_at": post["created_at"].isoformat(),
        "updated_at": post["updated_at"].isoformat()
    }


@app.post("/api/community/posts/{post_id}/comments")
async def add_comment(
    post_id: str,
    comment_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Add comment to post"""
    post = await db.community_posts.find_one({"_id": ObjectId(post_id)})
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    comment_text = comment_data.get("comment_text")
    if not comment_text:
        raise HTTPException(status_code=400, detail="Comment text is required")
    
    comment_doc = {
        "post_id": post_id,
        "user_id": str(current_user["_id"]),
        "user_name": current_user["name"],
        "comment_text": comment_text,
        "media_urls": comment_data.get("media_urls", []),
        "likes": 0,
        "created_at": datetime.utcnow()
    }
    
    result = await db.community_comments.insert_one(comment_doc)
    
    # Update post comment count
    await db.community_posts.update_one(
        {"_id": ObjectId(post_id)},
        {
            "$push": {"comments": str(result.inserted_id)},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    
    # Broadcast to WebSocket
    await manager.broadcast({
        "type": "new_comment",
        "post_id": post_id,
        "comment": {
            "id": str(result.inserted_id),
            "comment_text": comment_text,
            "user_name": current_user["name"],
            "created_at": datetime.utcnow().isoformat()
        }
    })
    
    return {
        "comment_id": str(result.inserted_id),
        "message": "Comment added successfully"
    }


@app.post("/api/community/posts/{post_id}/like")
async def like_post(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Like a post"""
    post = await db.community_posts.find_one({"_id": ObjectId(post_id)})
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Check if already liked
    existing_like = await db.post_likes.find_one({
        "post_id": post_id,
        "user_id": str(current_user["_id"])
    })
    
    if existing_like:
        # Unlike
        await db.post_likes.delete_one({"_id": existing_like["_id"]})
        await db.community_posts.update_one(
            {"_id": ObjectId(post_id)},
            {"$inc": {"likes": -1}}
        )
        return {"message": "Post unliked", "liked": False}
    else:
        # Like
        await db.post_likes.insert_one({
            "post_id": post_id,
            "user_id": str(current_user["_id"]),
            "created_at": datetime.utcnow()
        })
        await db.community_posts.update_one(
            {"_id": ObjectId(post_id)},
            {"$inc": {"likes": 1}}
        )
        return {"message": "Post liked", "liked": True}


@app.post("/api/community/posts/{post_id}/helpful")
async def mark_helpful(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark post as helpful"""
    await db.community_posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$inc": {"helpful_count": 1}}
    )
    
    return {"message": "Marked as helpful"}


@app.post("/api/community/posts/{post_id}/solve")
async def mark_solved(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark question as solved (only by post author)"""
    post = await db.community_posts.find_one({"_id": ObjectId(post_id)})
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if post["author_id"] != str(current_user["_id"]):
        raise HTTPException(
            status_code=403,
            detail="Only the post author can mark as solved"
        )
    
    await db.community_posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$set": {"is_solved": True, "updated_at": datetime.utcnow()}}
    )
    
    return {"message": "Post marked as solved"}


@app.post("/api/crop-photos/{photo_id}/share-to-community")
async def share_analysis_to_community(
    photo_id: str,
    share_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Share crop analysis to community forum with language support"""
    
    user_language = current_user.get("language_preference", "telugu")
    
    photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
    
    if not photo:
        error_msg = "ఫోటో కనుగొనబడలేదు" if user_language == "telugu" else "Photo not found"
        raise HTTPException(status_code=404, detail=error_msg)
    
    if photo["user_id"] != str(current_user["_id"]):
        error_msg = "అధికారం లేదు" if user_language == "telugu" else "Not authorized"
        raise HTTPException(status_code=403, detail=error_msg)
    
    # Create community post with analysis reference
    title = share_data.get("title")
    content = share_data.get("content")
    
    # Ensure image URL is in media_urls
    media_urls = share_data.get("media_urls", [])
    if photo["image_url"] not in media_urls:
        media_urls.insert(0, photo["image_url"])
    
    post_doc = {
        "title": title,
        "content_text": content,
        "media_urls": media_urls,
        "author_id": str(current_user["_id"]),
        "author_name": current_user["name"],
        "location": current_user.get("village"),
        "tags": share_data.get("tags", []),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "comments": [],
        "likes": 0,
        "views": 0,
        "helpful_count": 0,
        "analysis_reference": {
            "photo_id": photo_id,
            "image_url": photo["image_url"],
            "disease": photo.get("disease"),
            "confidence": photo.get("confidence_score"),
            "suggested_treatment": photo.get("suggested_treatment")
        },
        "is_question": True,
        "is_solved": False,
        "language": user_language  # Add language tag
    }
    
    result = await db.community_posts.insert_one(post_doc)
    
    # Update photo with community post reference
    await db.crop_photos.update_one(
        {"_id": ObjectId(photo_id)},
        {"$set": {"community_post_id": str(result.inserted_id)}}
    )
    
    # Broadcast to WebSocket
    await manager.broadcast({
        "type": "new_post",
        "post": {
            "id": str(result.inserted_id),
            "title": title,
            "author": current_user["name"],
            "has_analysis": True,
            "image_url": photo["image_url"],
            "language": user_language,
            "created_at": datetime.utcnow().isoformat()
        }
    })
    
    success_msg = "విశ్లేషణ విజయవంతంగా సంఘానికి పంచుకోబడింది" if user_language == "telugu" else "Analysis shared to community successfully"
    
    return {
        "post_id": str(result.inserted_id),
        "message": success_msg
    }

@app.get("/api/community/trending-topics")
async def get_trending_topics(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get trending topics/tags"""
    pipeline = [
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    
    trending = await db.community_posts.aggregate(pipeline).to_list(limit)
    
    return [
        {"tag": t["_id"], "count": t["count"]}
        for t in trending
    ]


@app.get("/api/community/my-posts")
async def get_my_posts(
    current_user: dict = Depends(get_current_user)
):
    """Get current user's posts"""
    posts = await db.community_posts.find({
        "author_id": str(current_user["_id"])
    }).sort("created_at", -1).to_list(100)
    
    result = []
    for post in posts:
        comments_count = await db.community_comments.count_documents(
            {"post_id": str(post["_id"])}
        )
        
        result.append({
            "id": str(post["_id"]),
            "title": post["title"],
            "is_solved": post.get("is_solved", False),
            "comments_count": comments_count,
            "likes": post.get("likes", 0),
            "views": post.get("views", 0),
            "created_at": post["created_at"].isoformat()
        })
    
    return result


@app.delete("/api/community/posts/{post_id}")
async def delete_post(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a post (only by author or admin)"""
    post = await db.community_posts.find_one({"_id": ObjectId(post_id)})
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if post["author_id"] != str(current_user["_id"]) and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Delete post and its comments
    await db.community_posts.delete_one({"_id": ObjectId(post_id)})
    await db.community_comments.delete_many({"post_id": post_id})
    
    return {"message": "Post deleted successfully"}
# ============= WebSocket Routes =============
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "message":
                message_data = {
                    "sender_id": data["sender_id"],
                    "receiver_id": data.get("receiver_id"),
                    "group_id": data.get("group_id"),
                    "message_text": data["message"],
                    "timestamp": datetime.utcnow()
                }
                await db.chat_messages.insert_one(message_data)
                await manager.broadcast(data)
            
            elif data["type"] == "typing":
                await manager.broadcast({"type": "typing", "user_id": data["user_id"]})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============= Weather Alerts =============
@app.get("/api/weather/alerts")
async def get_weather_alerts(current_user: dict = Depends(get_current_user)):
    location = current_user.get("district", "")
    alerts = await db.weather_alerts.find({"location": location}).sort("timestamp", -1).limit(10).to_list(10)
    
    return [
        {
            "id": str(a["_id"]),
            "type": a["alert_type"],
            "message": a["message"],
            "recommended_action": a["recommended_action"],
            "timestamp": a["timestamp"].isoformat()
        } for a in alerts
    ]


# ============= Advisory Calendar =============
@app.get("/api/advisory/calendar")
async def get_advisory_calendar(crop_id: str, current_user: dict = Depends(get_current_user)):
    advisories = await db.advisory_calendar.find({"crop_id": crop_id}).to_list(12)
    
    return [
        {
            "month": a["month"],
            "preventive_actions": a["preventive_actions"],
            "treatment_alerts": a["treatment_alerts"]
        } for a in advisories
    ]


# ============= Gamification Routes =============
@app.get("/api/gamification/badges")
async def get_user_badges(current_user: dict = Depends(get_current_user)):
    user_badges = await db.user_badges.find({"user_id": str(current_user["_id"])}).to_list(100)
    
    result = []
    for ub in user_badges:
        badge = await db.badges.find_one({"_id": ObjectId(ub["badge_id"])})
        if badge:
            result.append({
                "name": badge["name"],
                "description": badge["description"],
                "earned_at": ub["earned_at"].isoformat()
            })
    
    return result

@app.get("/api/gamification/leaderboard")
async def get_leaderboard(category: str = "overall"):
    leaderboard = await db.leaderboards.find({"category": category}).sort("score", -1).limit(10).to_list(10)
    
    result = []
    for entry in leaderboard:
        user = await db.users.find_one({"_id": ObjectId(entry["user_id"])})
        if user:
            result.append({
                "rank": entry["rank"],
                "name": user["name"],
                "score": entry["score"],
                "village": user.get("village")
            })
    
    return result

# Add these routes to your main.py file

# ============= Organic Solutions Routes =============
@app.post("/api/organic-solutions/upload-image")
async def upload_solution_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload image for organic solution"""
    
    if current_user["role"] not in ["specialist", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Only specialists can upload solution images"
        )
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG, PNG, and JPG images are allowed"
        )
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum size is 10MB"
        )
    
    try:
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            contents,
            folder="organic_solutions",
            resource_type="image",
            transformation=[
                {'width': 1200, 'height': 1200, 'crop': 'limit'},
                {'quality': 'auto'}
            ]
        )
        
        return {
            "image_url": upload_result.get("secure_url"),
            "message": "Image uploaded successfully"
        }
        
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload image"
        )
@app.get("/api/organic-solutions/{solution_id}")
async def get_solution_detail(
    solution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific solution"""
    solution = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
    
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    
    # Get average rating
    ratings_pipeline = [
        {"$match": {"solution_id": solution_id}},
        {"$group": {
            "_id": None,
            "avg_rating": {"$avg": "$rating"},
            "total_ratings": {"$sum": 1}
        }}
    ]
    rating_result = await db.solution_ratings.aggregate(ratings_pipeline).to_list(1)
    
    avg_rating = rating_result[0]["avg_rating"] if rating_result else 0
    total_ratings = rating_result[0]["total_ratings"] if rating_result else 0
    
    # Get application count
    application_count = await db.solution_applications.count_documents({
        "solution_id": solution_id
    })
    
    return {
        "id": str(solution["_id"]),
        "title": solution["title"],
        "description": solution["description"],
        "category": solution["category"],
        "success_rate": solution["success_rate"],
        "cost_per_acre": solution["cost_per_acre"],
        "preparation_time": solution["preparation_time"],
        "ingredients": solution.get("ingredients", []),
        "preparation_steps": solution.get("preparation_steps", []),
        "application_method": solution["application_method"],
        "application_frequency": solution["application_frequency"],
        "diseases_treated": solution.get("diseases_treated", []),
        "crops_suitable_for": solution.get("crops_suitable_for", []),
        "precautions": solution.get("precautions", []),
        "local_names": solution.get("local_names", {}),
        "seasonal_effectiveness": solution.get("seasonal_effectiveness", {}),
        "media_urls": solution.get("media_urls", []),
        "average_rating": round(avg_rating, 1) if avg_rating else 0,
        "total_ratings": total_ratings,
        "applications_count": application_count,
        "created_by": solution.get("created_by"),  # ✅ Added this
        "created_at": solution["created_at"].isoformat()
    }

@app.post("/api/organic-solutions")
async def create_organic_solution(
    solution_data: dict,  # Changed from SolutionCreateRequest to accept image_url
    current_user: dict = Depends(get_current_user)
):
    """Create a new organic solution with image"""
    if current_user["role"] not in ["specialist", "admin"]:
        raise HTTPException(
            status_code=403, 
            detail="Only specialists can create solutions"
        )
    
    solution_doc = {
        "title": solution_data["title"],
        "description": solution_data["description"],
        "category": solution_data["category"],
        "success_rate": solution_data.get("success_rate", 0.0),
        "cost_per_acre": solution_data.get("cost_per_acre", 0.0),
        "preparation_time": solution_data["preparation_time"],
        "ingredients": solution_data.get("ingredients", []),
        "preparation_steps": solution_data.get("preparation_steps", []),
        "application_method": solution_data["application_method"],
        "application_frequency": solution_data["application_frequency"],
        "diseases_treated": solution_data.get("diseases_treated", []),
        "crops_suitable_for": solution_data.get("crops_suitable_for", []),
        "precautions": solution_data.get("precautions", []),
        "local_names": solution_data.get("local_names", {}),
        "seasonal_effectiveness": solution_data.get("seasonal_effectiveness", {}),
        
        # NEW: Image fields
        "image_url": solution_data.get("image_url"),
        "media_urls": solution_data.get("media_urls", []),
        
        "created_by": str(current_user["_id"]),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.organic_solutions.insert_one(solution_doc)
    
    return {
        "solution_id": str(result.inserted_id),
        "message": "Organic solution created successfully"
    }
@app.get("/api/organic-solutions")
async def get_organic_solutions(
    category: Optional[str] = None,
    disease: Optional[str] = None,
    crop: Optional[str] = None,
    min_success_rate: Optional[float] = None,
    max_cost: Optional[float] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get list of organic solutions with filters"""
    query = {}
    
    if category:
        query["category"] = category
    
    if disease:
        query["diseases_treated"] = {"$in": [disease]}
    
    if crop:
        query["crops_suitable_for"] = {"$in": [crop]}
    
    if min_success_rate is not None:
        query["success_rate"] = {"$gte": min_success_rate}
    
    if max_cost is not None:
        if "cost_per_acre" not in query:
            query["cost_per_acre"] = {}
        query["cost_per_acre"]["$lte"] = max_cost
    
    solutions = await db.organic_solutions.find(query).sort(
        "success_rate", -1
    ).limit(limit).to_list(limit)
    
    return [
        {
            "id": str(s["_id"]),
            "title": s["title"],
            "description": s["description"],
            "category": s["category"],
            "success_rate": s["success_rate"],
            "cost_per_acre": s["cost_per_acre"],
            "preparation_time": s["preparation_time"],
            "application_method": s["application_method"],
            "application_frequency": s.get("application_frequency", ""),  # Added
            "image_url": s.get("image_url"),  # THIS WAS MISSING - ADD THIS LINE
            "ingredients": s.get("ingredients", []),  # Added
            "preparation_steps": s.get("preparation_steps", []),  # Added
            "diseases_treated": s.get("diseases_treated", []),
            "crops_suitable_for": s.get("crops_suitable_for", []),
            "precautions": s.get("precautions", []),  # Added
            "average_rating": s.get("average_rating", 0),  # Added
            "applications_count": s.get("applications_count", 0),  # Added
            "created_at": s["created_at"].isoformat() if "created_at" in s else None
        } for s in solutions
    ]


@app.post("/api/organic-solutions/{solution_id}/apply")
async def apply_organic_solution(
    solution_id: str,
    application_data: dict,  # This should accept the data
    current_user: dict = Depends(get_current_user)
):
    """Record application of an organic solution"""
    solution = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
    
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    
    application_doc = {
        "solution_id": solution_id,
        "user_id": str(current_user["_id"]),
        "crop_id": application_data.get("crop_id"),
        "photo_id": application_data.get("photo_id"),
        "area_applied": application_data.get("area_applied", 1.0),  # Default 1 acre
        "location": application_data.get("location") or current_user.get("village"),
        "notes": application_data.get("notes"),
        "before_photo_url": application_data.get("before_photo_url"),
        "status": "applied",
        "applied_at": datetime.utcnow(),
        "follow_up_dates": []
    }
    
    result = await db.solution_applications.insert_one(application_doc)
    
    # THIS IS CRITICAL - Update user progress
    await db.user_progress.update_one(
        {"user_id": str(current_user["_id"])},
        {
            "$inc": {"treatments_applied": 1},
            "$set": {"last_updated": datetime.utcnow()}
        },
        upsert=True
    )
    
    return {
        "application_id": str(result.inserted_id),
        "message": "Solution application recorded successfully"
    }

@app.get("/api/organic-solutions/{solution_id}")
async def get_solution_detail(
    solution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific solution"""
    solution = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
    
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    
    # ... existing rating code ...
    
    return {
        "id": str(solution["_id"]),
        "title": solution["title"],
        "description": solution["description"],
        "category": solution["category"],
        "success_rate": solution["success_rate"],
        "cost_per_acre": solution["cost_per_acre"],
        "preparation_time": solution["preparation_time"],
        "ingredients": solution.get("ingredients", []),
        "preparation_steps": solution.get("preparation_steps", []),
        "application_method": solution["application_method"],
        "application_frequency": solution["application_frequency"],
        "diseases_treated": solution.get("diseases_treated", []),
        "crops_suitable_for": solution.get("crops_suitable_for", []),
        "precautions": solution.get("precautions", []),
        "local_names": solution.get("local_names", {}),
        "seasonal_effectiveness": solution.get("seasonal_effectiveness", {}),
        
        # Images
        "image_url": solution.get("image_url"),
        "media_urls": solution.get("media_urls", []),
        
        "average_rating": round(avg_rating, 1) if avg_rating else 0,
        "total_ratings": total_ratings,
        "applications_count": application_count,
        "created_by": solution.get("created_by"),
        "created_at": solution["created_at"].isoformat()
    }
# Fixed Update Organic Solution Endpoint


@app.put("/api/organic-solutions/{solution_id}")
async def update_organic_solution(
    solution_id: str,
    solution_data: dict,
    current_user: dict = Depends(require_role("specialist"))
):
    """Update an organic solution - Specialist only, or admin can update any"""
    try:
        # Validate ObjectId
        if not ObjectId.is_valid(solution_id):
            raise HTTPException(status_code=400, detail="Invalid solution ID")
        
        # Get existing solution
        solution = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Check ownership - allow if user is admin, or if user created it, or if created_by doesn't exist (legacy data)
        if current_user["role"] != "admin":
            # Check if created_by exists and matches current user
            if "created_by" in solution and solution["created_by"] != str(current_user["_id"]):
                raise HTTPException(
                    status_code=403, 
                    detail="You can only update your own solutions"
                )
        
        # Prepare update data
        update_data = {
            "title": solution_data.get("title"),
            "description": solution_data.get("description"),
            "category": solution_data.get("category"),
            "success_rate": solution_data.get("success_rate"),
            "cost_per_acre": solution_data.get("cost_per_acre"),
            "preparation_time": solution_data.get("preparation_time"),
            "application_method": solution_data.get("application_method"),
            "application_frequency": solution_data.get("application_frequency"),
            "image_url": solution_data.get("image_url"),
            "ingredients": solution_data.get("ingredients", []),
            "preparation_steps": solution_data.get("preparation_steps", []),
            "diseases_treated": solution_data.get("diseases_treated", []),
            "crops_suitable_for": solution_data.get("crops_suitable_for", []),
            "precautions": solution_data.get("precautions", []),
            "updated_at": datetime.utcnow()
        }
        
        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}
        
        # If created_by doesn't exist, add it now
        if "created_by" not in solution:
            update_data["created_by"] = str(current_user["_id"])
        
        # Update in database
        result = await db.organic_solutions.update_one(
            {"_id": ObjectId(solution_id)},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            # Check if document exists but no changes were made
            existing = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
            if existing:
                return JSONResponse(
                    content={"message": "No changes detected", "id": solution_id},
                    status_code=200
                )
            raise HTTPException(status_code=404, detail="Solution not found")
        
        return {"message": "Solution updated successfully", "id": solution_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating solution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/organic-solutions/{solution_id}")
async def delete_organic_solution(
    solution_id: str,
    current_user: dict = Depends(require_role("specialist"))
):
    """Delete an organic solution - Specialist only, or admin can delete any"""
    try:
        # Validate ObjectId
        if not ObjectId.is_valid(solution_id):
            raise HTTPException(status_code=400, detail="Invalid solution ID")
        
        # Get existing solution
        solution = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        # Check ownership - allow if user is admin, or if user created it, or if created_by doesn't exist (legacy data)
        if current_user["role"] != "admin":
            # Check if created_by exists and matches current user
            if "created_by" in solution and solution["created_by"] != str(current_user["_id"]):
                raise HTTPException(
                    status_code=403, 
                    detail="You can only delete your own solutions"
                )
        
        # Delete the solution
        result = await db.organic_solutions.delete_one({"_id": ObjectId(solution_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        return {"message": "Solution deleted successfully", "id": solution_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting solution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/organic-solutions/{solution_id}/apply")
async def apply_organic_solution(
    solution_id: str,
    application_data: ApplicationCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Record application of an organic solution"""
    solution = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
    
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    
    application_doc = {
        "solution_id": solution_id,
        "user_id": str(current_user["_id"]),
        "crop_id": application_data.crop_id,
        "photo_id": application_data.photo_id,
        "area_applied": application_data.area_applied,
        "location": application_data.location or current_user.get("village"),
        "notes": application_data.notes,
        "before_photo_url": application_data.before_photo_url,
        "status": "applied",
        "applied_at": datetime.utcnow(),
        "follow_up_dates": []
    }
    
    result = await db.solution_applications.insert_one(application_doc)
    
    return {
        "application_id": str(result.inserted_id),
        "message": "Solution application recorded successfully"
    }


@app.get("/api/organic-solutions/{solution_id}/applications")
async def get_solution_applications(
    solution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get applications of a specific solution"""
    applications = await db.solution_applications.find({
        "solution_id": solution_id,
        "user_id": str(current_user["_id"])
    }).sort("applied_at", -1).to_list(100)
    
    return [
        {
            "id": str(a["_id"]),
            "area_applied": a.get("area_applied"),
            "location": a.get("location"),
            "status": a["status"],
            "outcome": a.get("outcome"),
            "notes": a.get("notes"),
            "before_photo_url": a.get("before_photo_url"),
            "after_photo_url": a.get("after_photo_url"),
            "applied_at": a["applied_at"].isoformat(),
            "completed_at": a["completed_at"].isoformat() if a.get("completed_at") else None
        } for a in applications
    ]


@app.put("/api/solution-applications/{application_id}")
async def update_solution_application(
    application_id: str,
    update_data: ApplicationUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update solution application status"""
    application = await db.solution_applications.find_one({
        "_id": ObjectId(application_id)
    })
    
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    if application["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_fields = update_data.dict(exclude_unset=True)
    update_fields["updated_at"] = datetime.utcnow()
    
    if update_data.status == "completed":
        update_fields["completed_at"] = datetime.utcnow()
    
    await db.solution_applications.update_one(
        {"_id": ObjectId(application_id)},
        {"$set": update_fields}
    )
    
    return {"message": "Application updated successfully"}


@app.post("/api/organic-solutions/{solution_id}/rate")
async def rate_organic_solution(
    solution_id: str,
    rating_data: RatingCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Rate and review an organic solution"""
    solution = await db.organic_solutions.find_one({"_id": ObjectId(solution_id)})
    
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    
    # Check if user already rated
    existing_rating = await db.solution_ratings.find_one({
        "solution_id": solution_id,
        "user_id": str(current_user["_id"])
    })
    
    rating_doc = {
        "solution_id": solution_id,
        "user_id": str(current_user["_id"]),
        "rating": rating_data.rating,
        "review": rating_data.review,
        "effectiveness": rating_data.effectiveness,
        "ease_of_preparation": rating_data.ease_of_preparation,
        "cost_effectiveness": rating_data.cost_effectiveness,
        "would_recommend": rating_data.would_recommend,
        "created_at": datetime.utcnow()
    }
    
    if existing_rating:
        # Update existing rating
        rating_doc["updated_at"] = datetime.utcnow()
        await db.solution_ratings.update_one(
            {"_id": existing_rating["_id"]},
            {"$set": rating_doc}
        )
        return {"message": "Rating updated successfully"}
    else:
        # Create new rating
        result = await db.solution_ratings.insert_one(rating_doc)
        return {
            "rating_id": str(result.inserted_id),
            "message": "Rating submitted successfully"
        }


@app.get("/api/organic-solutions/{solution_id}/ratings")
async def get_solution_ratings(
    solution_id: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get ratings and reviews for a solution"""
    ratings = await db.solution_ratings.find({
        "solution_id": solution_id
    }).sort("created_at", -1).limit(limit).to_list(limit)
    
    result = []
    for r in ratings:
        user = await db.users.find_one({"_id": ObjectId(r["user_id"])})
        result.append({
            "rating": r["rating"],
            "review": r.get("review"),
            "effectiveness": r.get("effectiveness"),
            "ease_of_preparation": r.get("ease_of_preparation"),
            "cost_effectiveness": r.get("cost_effectiveness"),
            "would_recommend": r.get("would_recommend", True),
            "user_name": user["name"] if user else "Anonymous",
            "user_location": user.get("village") if user else None,
            "created_at": r["created_at"].isoformat()
        })
    
    return result

@app.get("/api/organic-solutions/search")
async def search_organic_solutions(
    query: str,
    current_user: dict = Depends(get_current_user)
):
    """Search organic solutions by keyword"""
    search_query = {
        "$or": [
            {"title": {"$regex": query, "$options": "i"}},
            {"description": {"$regex": query, "$options": "i"}},
            {"diseases_treated": {"$regex": query, "$options": "i"}},
            {"crops_suitable_for": {"$regex": query, "$options": "i"}},
            {"ingredients.name": {"$regex": query, "$options": "i"}}
        ]
    }
    
    solutions = await db.organic_solutions.find(search_query).limit(20).to_list(20)
    
    return [
        {
            "id": str(s["_id"]),
            "title": s["title"],
            "description": s["description"],
            "success_rate": s["success_rate"],
            "cost_per_acre": s["cost_per_acre"]
        } for s in solutions
    ]


from datetime import datetime

async def seed_organic_solutions():
    """చిత్రాలతో ప్రారంభ ఆర్గానిక్ పరిష్కారాల డేటాను సీడ్ చేయడం"""
    
    specialist_user = await db.users.find_one({"role": "specialist"})
    created_by_id = str(specialist_user["_id"]) if specialist_user else "000000000000000000000000"

    solutions_data = [
        {
            "title": "వేప నూనె స్ప్రే",
            "description": "వేప నూనె సహజ కీటకనాశకం. ఇది ఆఫిడ్స్, వైట్‌ఫ్లైస్, మీలీబగ్స్ వంటి హానికర కీటకాలను నియంత్రిస్తుంది. సరైన రీతిలో వాడితే మంచికీటకాలకు హాని ఉండదు.",
            "category": "పెస్టిసైడ్",
            "success_rate": 85.0,
            "cost_per_acre": 150.0,
            "preparation_time": "30 నిమిషాలు",
            "image_url": "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=800",
            "ingredients": [
                {"name": "వేప నూనె", "quantity": "10ml", "local_name": "వేప నూనె", "availability": "సులభంగా లభిస్తుంది"},
                {"name": "నీరు", "quantity": "1 లీటర్", "local_name": "నీరు", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "ద్రవ సబ్బు", "quantity": "2-3 చుక్కలు", "local_name": "ద్రవ సబ్బు", "availability": "సాధారణంగా లభిస్తుంది"}
            ],
            "preparation_steps": [
                "10ml వేప నూనెను 1 లీటర్ నీటిలో కలపండి.",
                "2-3 చుక్కల ద్రవ సబ్బు వేసి బాగా కలపండి.",
                "స్ప్రే బాటిల్‌లో పోసి వాడండి.",
                "తక్షణమే వాడితే ఎక్కువ ప్రభావం ఉంటుంది."
            ],
            "application_method": "మొక్కల మీద రెండు వైపులా పత్రాలపై స్ప్రే చేయండి.",
            "application_frequency": "ప్రతి 7-10 రోజులకు ఒకసారి",
            "diseases_treated": ["ఆఫిడ్స్", "వైట్‌ఫ్లైస్", "మీలీబగ్స్", "స్పైడర్ మైట్స్"],
            "crops_suitable_for": ["కూరగాయలు", "పత్తి", "మిరప", "టమాటా"],
            "precautions": [
                "ఉదయం లేదా సాయంత్రం వేళల్లో వాడండి.",
                "నేరుగా సూర్యకాంతిలో స్ప్రే చేయవద్దు.",
                "చిన్న మొక్కలపై ముందుగా పరీక్షించండి."
            ],
            "local_names": {"telugu": "వేప నూనె స్ప్రే", "hindi": "नीम का तेल स्प्रे"},
            "seasonal_effectiveness": {"Kharif": "అత్యంత ప్రభావవంతం", "Rabi": "సమర్థవంతం", "Summer": "మధ్యస్థంగా ప్రభావవంతం"},
            "media_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "పంచగవ్య",
            "description": "ఐదు ఆవు ఉత్పత్తులతో తయారైన సంప్రదాయ వృద్ధి ప్రేరక ద్రావణం. మొక్కల వృద్ధిని పెంచి, రోగనిరోధక శక్తిని మెరుగుపరుస్తుంది.",
            "category": "వృద్ధి ప్రేరక",
            "success_rate": 90.0,
            "cost_per_acre": 100.0,
            "preparation_time": "21 రోజులు",
            "image_url": "https://images.unsplash.com/photo-1500595046743-cd271d694d30?w=800",
            "ingredients": [
                {"name": "ఆవు పేడ", "quantity": "5 కిలోలు", "local_name": "ఆవు పేడ", "availability": "సులభంగా లభిస్తుంది"},
                {"name": "ఆవు మూత్రం", "quantity": "3 లీటర్లు", "local_name": "ఆవు మూత్రం", "availability": "సులభంగా లభిస్తుంది"},
                {"name": "పాలు", "quantity": "2 లీటర్లు", "local_name": "పాలు", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "నెయ్యి", "quantity": "1 కిలో", "local_name": "నెయ్యి", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "పెరుగు", "quantity": "2 లీటర్లు", "local_name": "పెరుగు", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "బెల్లం", "quantity": "3 కిలోలు", "local_name": "బెల్లం", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "అరటిపండు", "quantity": "12 పండ్లు", "local_name": "అరటిపండు", "availability": "సాధారణంగా లభిస్తుంది"}
            ],
            "preparation_steps": [
                "ఆవు పేడ మరియు నెయ్యి కలిపి మూడు రోజుల పాటు ఉంచండి.",
                "తరువాత ఆవు మూత్రం, పాలు, పెరుగు, బెల్లం, అరటిపండు కలపండి.",
                "రోజూ రెండు సార్లు కలుపుతూ 21 రోజుల పాటు నీడలో ఉంచండి.",
                "తరువాత ద్రావణాన్ని వడకట్టండి."
            ],
            "application_method": "30ml/1 లీటర్ నీటిగా dilute చేసి మొక్కలపై స్ప్రే చేయండి.",
            "application_frequency": "ప్రతి 15 రోజులకు ఒకసారి",
            "diseases_treated": ["మొక్కల సాధారణ ఆరోగ్యం", "రోగ నిరోధకత పెరుగుతుంది"],
            "crops_suitable_for": ["అన్ని పంటలు"],
            "precautions": [
                "దేశీ ఆవు ఉత్పత్తులు మాత్రమే వాడండి.",
                "చల్లని ప్రదేశంలో నిల్వ చేయండి.",
                "6 నెలల్లో వాడాలి."
            ],
            "local_names": {"telugu": "పంచగవ్య", "hindi": "पंचगव्य"},
            "seasonal_effectiveness": {"All": "అత్యంత ప్రభావవంతం"},
            "media_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "వర్మీకంపోస్ట్ టీ",
            "description": "వర్మీకంపోస్ట్ నుండి తయారయ్యే ద్రవ ఎరువు. ఇది సూక్ష్మజీవులు మరియు పోషకాలతో నిండిన సహజ ఎరువు.",
            "category": "ఎరువు",
            "success_rate": 88.0,
            "cost_per_acre": 80.0,
            "preparation_time": "24 గంటలు",
            "image_url": "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=800",
            "ingredients": [
                {"name": "వర్మీకంపోస్ట్", "quantity": "1 కిలో", "local_name": "వర్మీకంపోస్ట్", "availability": "ఆర్గానిక్ దుకాణాలలో లభిస్తుంది"},
                {"name": "నీరు", "quantity": "10 లీటర్లు", "local_name": "నీరు", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "బెల్లం", "quantity": "100 గ్రాములు", "local_name": "బెల్లం", "availability": "సాధారణంగా లభిస్తుంది"}
            ],
            "preparation_steps": [
                "10 లీటర్ల నీటిలో వర్మీకంపోస్ట్ ని వస్త్ర సంచిలో వేసి నానబెట్టండి.",
                "బెల్లం వేసి బాగా కలపండి.",
                "24 గంటలు ఉంచండి.",
                "తరువాత ద్రావణం వడకట్టండి."
            ],
            "application_method": "1:10 నీటిలో dilute చేసి మట్టి చుట్టూ పోయాలి లేదా స్ప్రే చేయాలి.",
            "application_frequency": "ప్రతి 10-15 రోజులకు ఒకసారి",
            "diseases_treated": ["పోషక లోపాలు", "మట్టి ఆరోగ్య మెరుగుదల"],
            "crops_suitable_for": ["అన్ని పంటలు"],
            "precautions": [
                "48 గంటల్లో వాడండి.",
                "చల్లని సమయాల్లో మాత్రమే వాడండి."
            ],
            "local_names": {"telugu": "వర్మీకంపోస్ట్ టీ", "hindi": "वर्मीकंपोस्ट चाय"},
            "seasonal_effectiveness": {"All": "సమర్థవంతం"},
            "media_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "జీవామృతం",
            "description": "మట్టి సూక్ష్మజీవులను పెంచే ద్రవ సేంద్రియ ఎరువు. ఇది పంటలకు జీవం అందిస్తుంది.",
            "category": "ఎరువు",
            "success_rate": 87.0,
            "cost_per_acre": 50.0,
            "preparation_time": "7 రోజులు",
            "image_url": "https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=800",
            "ingredients": [
                {"name": "ఆవు పేడ", "quantity": "10 కిలోలు", "local_name": "ఆవు పేడ", "availability": "సులభంగా లభిస్తుంది"},
                {"name": "ఆవు మూత్రం", "quantity": "10 లీటర్లు", "local_name": "ఆవు మూత్రం", "availability": "సులభంగా లభిస్తుంది"},
                {"name": "బెల్లం", "quantity": "2 కిలోలు", "local_name": "బెల్లం", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "పప్పు పిండి", "quantity": "2 కిలోలు", "local_name": "పప్పు పిండి", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "అడవి మట్టి", "quantity": "1 ముట్టు", "local_name": "అడవి మట్టి", "availability": "ప్రాంతంలో లభిస్తుంది"}
            ],
            "preparation_steps": [
                "ఆవు పేడ, నీరు కలపండి.",
                "ఆవు మూత్రం, బెల్లం, పప్పు పిండి కలిపి జోడించండి.",
                "అడవి మట్టి వేసి బాగా కలపండి.",
                "7 రోజుల పాటు నీడలో ఉంచి రోజూ రెండు సార్లు కలపండి."
            ],
            "application_method": "1:10 నీటిలో dilute చేసి మొక్కల వద్ద పోయండి.",
            "application_frequency": "ప్రతి 15 రోజులకు ఒకసారి",
            "diseases_treated": ["మట్టి ఆరోగ్యం", "సూక్ష్మజీవాల పెంపు"],
            "crops_suitable_for": ["అన్ని పంటలు"],
            "precautions": [
                "తాజా ఆవు ఉత్పత్తులు వాడండి.",
                "7 రోజుల్లో వాడండి."
            ],
            "local_names": {"telugu": "జీవామృతం", "hindi": "जीवामृत"},
            "seasonal_effectiveness": {"All": "అత్యంత ప్రభావవంతం"},
            "media_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "వెల్లుల్లి మిర్చి స్ప్రే",
            "description": "సహజ కీటకనాశకం. ఇది ఆఫిడ్స్, త్రిప్స్, కేటర్పిల్లర్స్ వంటి కీటకాలను సమర్థంగా నియంత్రిస్తుంది.",
            "category": "పెస్టిసైడ్",
            "success_rate": 80.0,
            "cost_per_acre": 120.0,
            "preparation_time": "2 గంటలు",
            "image_url": "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?w=800",
            "ingredients": [
                {"name": "వెల్లుల్లి", "quantity": "100 గ్రాములు", "local_name": "వెల్లుల్లి", "availability": "సులభంగా లభిస్తుంది"},
                {"name": "పచ్చిమిర్చి", "quantity": "100 గ్రాములు", "local_name": "పచ్చిమిర్చి", "availability": "సులభంగా లభిస్తుంది"},
                {"name": "నీరు", "quantity": "2 లీటర్లు", "local_name": "నీరు", "availability": "సాధారణంగా లభిస్తుంది"},
                {"name": "సబ్బు ద్రావణం", "quantity": "10ml", "local_name": "సబ్బు", "availability": "సాధారణంగా లభిస్తుంది"}
            ],
            "preparation_steps": [
                "వెల్లుల్లి, పచ్చిమిర్చి కలిపి ముద్దలా చేయండి.",
                "1 లీటర్ నీటిలో రాత్రంతా నానబెట్టండి.",
                "తరువాత 30 నిమిషాలు మరిగించండి.",
                "చల్లారిన తర్వాత వడకట్టండి.",
                "సబ్బు ద్రావణం వేసి బాగా కలపండి."
            ],
            "application_method": "మొక్కలపై సమానంగా స్ప్రే చేయండి.",
            "application_frequency": "ప్రతి 5-7 రోజులకు ఒకసారి",
            "diseases_treated": ["ఆఫిడ్స్", "త్రిప్స్", "కేటర్పిల్లర్స్"],
            "crops_suitable_for": ["కూరగాయలు", "పత్తి", "మిరప"],
            "precautions": [
                "కళ్లలో పడకుండా జాగ్రత్త.",
                "సాయంత్రం వేళల్లో స్ప్రే చేయండి."
            ],
            "local_names": {"telugu": "వెల్లుల్లి మిర్చి స్ప్రే", "hindi": "लहसुन मिर्च स्प्रे"},
            "seasonal_effectiveness": {"All": "సమర్థవంతం"},
            "media_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]

    try:
        existing_count = await db.organic_solutions.count_documents({})
        if existing_count > 0:
            print(f"ఆర్గానిక్ పరిష్కారాలు ఇప్పటికే ఉన్నాయి ({existing_count} రికార్డులు). సీడ్ చేయడం మానివేశారు.")
            return
        
        result = await db.organic_solutions.insert_many(solutions_data)
        print(f"{len(result.inserted_ids)} ఆర్గానిక్ పరిష్కారాలు విజయవంతంగా సీడ్ అయ్యాయి!")
        
    except Exception as e:
        print(f"ఆర్గానిక్ పరిష్కారాలను సీడ్ చేయడంలో లోపం: {e}")

@app.post("/api/admin/seed-organic-solutions")
async def trigger_seed_organic_solutions(
    current_user: dict = Depends(require_role("specialist"))
):
    """Seed organic solutions data - Specialist only"""
    await seed_organic_solutions()
    return {"message": "Organic solutions seeded successfully"}


#  Traditional Knowledge Integration

# Add these models to your models.py file

class TraditionalPractice(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description: str
    category: str  # planting, pest_control, soil_management, water_conservation, seed_treatment
    region: str  # Which tribal community or region
    tribe_name: Optional[str] = None
    local_language: Optional[str] = None
    best_for_crops: List[str] = []
    season: str  # all_seasons, kharif, rabi, summer
    implements_needed: List[str] = []
    duration: Optional[str] = None
    success_stories: List[Dict] = []  # [{farmer_name, location, result}]
    scientific_basis: Optional[str] = None  # Modern scientific explanation
    media_urls: List[str] = []
    video_urls: List[str] = []
    local_names: Dict[str, str] = {}  # {language: local_name}
    difficulty_level: str = "medium"  # easy, medium, hard
    created_by: Optional[str] = None
    verified_by_elders: bool = False
    elder_name: Optional[str] = None
    elder_contact: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class TraditionalPracticeApplication(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    practice_id: str
    user_id: str
    crop_id: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    status: str = "applied"  # applied, in_progress, completed
    outcome: Optional[str] = None  # success, partial, failed
    feedback: Optional[str] = None
    before_photo_url: Optional[str] = None
    after_photo_url: Optional[str] = None
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    results: Optional[Dict] = None  # {yield_increase, cost_saved, etc}

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class TraditionalPracticeRating(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    practice_id: str
    user_id: str
    rating: int  # 1-5
    review: Optional[str] = None
    ease_of_implementation: Optional[int] = None  # 1-5
    effectiveness: Optional[int] = None  # 1-5
    would_recommend: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request Models
class PracticeCreateRequest(BaseModel):
    title: str
    description: str
    category: str
    region: str
    tribe_name: Optional[str] = None
    local_language: Optional[str] = None
    best_for_crops: List[str] = []
    season: str = "all_seasons"
    implements_needed: List[str] = []
    duration: Optional[str] = None
    scientific_basis: Optional[str] = None
    local_names: Optional[Dict[str, str]] = {}
    difficulty_level: str = "medium"
    verified_by_elders: bool = False
    elder_name: Optional[str] = None
    elder_contact: Optional[str] = None


class PracticeUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    region: Optional[str] = None
    tribe_name: Optional[str] = None
    best_for_crops: Optional[List[str]] = None
    season: Optional[str] = None
    implements_needed: Optional[List[str]] = None
    duration: Optional[str] = None
    scientific_basis: Optional[str] = None
    difficulty_level: Optional[str] = None


class PracticeApplicationRequest(BaseModel):
    crop_id: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    before_photo_url: Optional[str] = None


class PracticeRatingRequest(BaseModel):
    rating: int
    review: Optional[str] = None
    ease_of_implementation: Optional[int] = None
    effectiveness: Optional[int] = None
    would_recommend: bool = True


# ============= Traditional Knowledge Routes =============
# Add these routes to your main.py

@app.post("/api/traditional-practices")
async def create_traditional_practice(
    practice_data: PracticeCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new traditional practice"""
    if current_user["role"] not in ["specialist", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Only specialists and admins can create practices"
        )
    
    practice_doc = {
        "title": practice_data.title,
        "description": practice_data.description,
        "category": practice_data.category,
        "region": practice_data.region,
        "tribe_name": practice_data.tribe_name,
        "local_language": practice_data.local_language,
        "best_for_crops": practice_data.best_for_crops,
        "season": practice_data.season,
        "implements_needed": practice_data.implements_needed,
        "duration": practice_data.duration,
        "scientific_basis": practice_data.scientific_basis,
        "local_names": practice_data.local_names or {},
        "difficulty_level": practice_data.difficulty_level,
        "verified_by_elders": practice_data.verified_by_elders,
        "elder_name": practice_data.elder_name,
        "elder_contact": practice_data.elder_contact,
        "success_stories": [],
        "media_urls": [],
        "video_urls": [],
        "created_by": str(current_user["_id"]),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.traditional_practices.insert_one(practice_doc)
    
    return {
        "practice_id": str(result.inserted_id),
        "message": "Traditional practice created successfully"
    }


@app.get("/api/traditional-practices")
async def get_traditional_practices(
    category: Optional[str] = None,
    region: Optional[str] = None,
    tribe: Optional[str] = None,
    season: Optional[str] = None,
    crop: Optional[str] = None,
    difficulty: Optional[str] = None,
    verified_only: bool = False,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get list of traditional practices with filters"""
    query = {}
    
    if category:
        query["category"] = category
    
    if region:
        query["region"] = {"$regex": region, "$options": "i"}
    
    if tribe:
        query["tribe_name"] = {"$regex": tribe, "$options": "i"}
    
    if season:
        query["season"] = season
    
    if crop:
        query["best_for_crops"] = {"$in": [crop]}
    
    if difficulty:
        query["difficulty_level"] = difficulty
    
    if verified_only:
        query["verified_by_elders"] = True
    
    practices = await db.traditional_practices.find(query).sort(
        "created_at", -1
    ).limit(limit).to_list(limit)
    
    return [
        {
            "id": str(p["_id"]),
            "title": p["title"],
            "description": p["description"],
            "category": p["category"],
            "region": p["region"],
            "tribe_name": p.get("tribe_name"),
            "best_for_crops": p.get("best_for_crops", []),
            "season": p["season"],
            "difficulty_level": p.get("difficulty_level", "medium"),
            "verified_by_elders": p.get("verified_by_elders", False),
            "elder_name": p.get("elder_name"),
            "created_at": p["created_at"].isoformat()
        } for p in practices
    ]


@app.get("/api/traditional-practices/{practice_id}")
async def get_practice_detail(
    practice_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific practice"""
    practice = await db.traditional_practices.find_one({"_id": ObjectId(practice_id)})
    
    if not practice:
        raise HTTPException(status_code=404, detail="Practice not found")
    
    # Get average rating
    ratings_pipeline = [
        {"$match": {"practice_id": practice_id}},
        {"$group": {
            "_id": None,
            "avg_rating": {"$avg": "$rating"},
            "avg_ease": {"$avg": "$ease_of_implementation"},
            "avg_effectiveness": {"$avg": "$effectiveness"},
            "total_ratings": {"$sum": 1}
        }}
    ]
    rating_result = await db.practice_ratings.aggregate(ratings_pipeline).to_list(1)
    
    avg_rating = rating_result[0]["avg_rating"] if rating_result else 0
    avg_ease = rating_result[0]["avg_ease"] if rating_result else 0
    avg_effectiveness = rating_result[0]["avg_effectiveness"] if rating_result else 0
    total_ratings = rating_result[0]["total_ratings"] if rating_result else 0
    
    # Get application count
    application_count = await db.practice_applications.count_documents({
        "practice_id": practice_id
    })
    
    # Get success rate
    success_count = await db.practice_applications.count_documents({
        "practice_id": practice_id,
        "outcome": "success"
    })
    success_rate = (success_count / application_count * 100) if application_count > 0 else 0
    
    return {
        "id": str(practice["_id"]),
        "title": practice["title"],
        "description": practice["description"],
        "category": practice["category"],
        "region": practice["region"],
        "tribe_name": practice.get("tribe_name"),
        "local_language": practice.get("local_language"),
        "best_for_crops": practice.get("best_for_crops", []),
        "season": practice["season"],
        "implements_needed": practice.get("implements_needed", []),
        "duration": practice.get("duration"),
        "scientific_basis": practice.get("scientific_basis"),
        "local_names": practice.get("local_names", {}),
        "difficulty_level": practice.get("difficulty_level", "medium"),
        "verified_by_elders": practice.get("verified_by_elders", False),
        "elder_name": practice.get("elder_name"),
        "elder_contact": practice.get("elder_contact"),
        "success_stories": practice.get("success_stories", []),
        "media_urls": practice.get("media_urls", []),
        "video_urls": practice.get("video_urls", []),
        "average_rating": round(avg_rating, 1),
        "average_ease": round(avg_ease, 1),
        "average_effectiveness": round(avg_effectiveness, 1),
        "total_ratings": total_ratings,
        "applications_count": application_count,
        "success_rate": round(success_rate, 1),
        "created_at": practice["created_at"].isoformat()
    }


@app.put("/api/traditional-practices/{practice_id}")
async def update_traditional_practice(
    practice_id: str,
    update_data: PracticeUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a traditional practice"""
    practice = await db.traditional_practices.find_one({"_id": ObjectId(practice_id)})
    
    if not practice:
        raise HTTPException(status_code=404, detail="Practice not found")
    
    if practice["created_by"] != str(current_user["_id"]) and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to update this practice")
    
    update_fields = update_data.dict(exclude_unset=True)
    update_fields["updated_at"] = datetime.utcnow()
    
    await db.traditional_practices.update_one(
        {"_id": ObjectId(practice_id)},
        {"$set": update_fields}
    )
    
    return {"message": "Practice updated successfully"}


@app.delete("/api/traditional-practices/{practice_id}")
async def delete_traditional_practice(
    practice_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a traditional practice"""
    practice = await db.traditional_practices.find_one({"_id": ObjectId(practice_id)})
    
    if not practice:
        raise HTTPException(status_code=404, detail="Practice not found")
    
    if practice["created_by"] != str(current_user["_id"]) and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this practice")
    
    await db.traditional_practices.delete_one({"_id": ObjectId(practice_id)})
    
    return {"message": "Practice deleted successfully"}


@app.post("/api/traditional-practices/{practice_id}/apply")
async def apply_traditional_practice(
    practice_id: str,
    application_data: PracticeApplicationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Record application of a traditional practice"""
    practice = await db.traditional_practices.find_one({"_id": ObjectId(practice_id)})
    
    if not practice:
        raise HTTPException(status_code=404, detail="Practice not found")
    
    application_doc = {
        "practice_id": practice_id,
        "user_id": str(current_user["_id"]),
        "crop_id": application_data.crop_id,
        "location": application_data.location or current_user.get("village"),
        "notes": application_data.notes,
        "before_photo_url": application_data.before_photo_url,
        "status": "applied",
        "applied_at": datetime.utcnow()
    }
    
    result = await db.practice_applications.insert_one(application_doc)
    
    return {
        "application_id": str(result.inserted_id),
        "message": "Practice application recorded successfully"
    }


@app.get("/api/traditional-practices/{practice_id}/applications")
async def get_practice_applications(
    practice_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get applications of a specific practice"""
    applications = await db.practice_applications.find({
        "practice_id": practice_id,
        "user_id": str(current_user["_id"])
    }).sort("applied_at", -1).to_list(100)
    
    return [
        {
            "id": str(a["_id"]),
            "location": a.get("location"),
            "status": a["status"],
            "outcome": a.get("outcome"),
            "notes": a.get("notes"),
            "before_photo_url": a.get("before_photo_url"),
            "after_photo_url": a.get("after_photo_url"),
            "applied_at": a["applied_at"].isoformat(),
            "completed_at": a["completed_at"].isoformat() if a.get("completed_at") else None
        } for a in applications
    ]


@app.put("/api/practice-applications/{application_id}")
async def update_practice_application(
    application_id: str,
    status: Optional[str] = None,
    outcome: Optional[str] = None,
    notes: Optional[str] = None,
    after_photo_url: Optional[str] = None,
    feedback: Optional[str] = None,
    results: Optional[Dict] = None,
    current_user: dict = Depends(get_current_user)
):
    """Update practice application status"""
    application = await db.practice_applications.find_one({
        "_id": ObjectId(application_id)
    })
    
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    if application["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_fields = {}
    if status:
        update_fields["status"] = status
    if outcome:
        update_fields["outcome"] = outcome
    if notes:
        update_fields["notes"] = notes
    if after_photo_url:
        update_fields["after_photo_url"] = after_photo_url
    if feedback:
        update_fields["feedback"] = feedback
    if results:
        update_fields["results"] = results
    
    if status == "completed":
        update_fields["completed_at"] = datetime.utcnow()
    
    await db.practice_applications.update_one(
        {"_id": ObjectId(application_id)},
        {"$set": update_fields}
    )
    
    return {"message": "Application updated successfully"}


@app.post("/api/traditional-practices/{practice_id}/rate")
async def rate_traditional_practice(
    practice_id: str,
    rating_data: PracticeRatingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Rate and review a traditional practice"""
    practice = await db.traditional_practices.find_one({"_id": ObjectId(practice_id)})
    
    if not practice:
        raise HTTPException(status_code=404, detail="Practice not found")
    
    # Check if user already rated
    existing_rating = await db.practice_ratings.find_one({
        "practice_id": practice_id,
        "user_id": str(current_user["_id"])
    })
    
    rating_doc = {
        "practice_id": practice_id,
        "user_id": str(current_user["_id"]),
        "rating": rating_data.rating,
        "review": rating_data.review,
        "ease_of_implementation": rating_data.ease_of_implementation,
        "effectiveness": rating_data.effectiveness,
        "would_recommend": rating_data.would_recommend,
        "created_at": datetime.utcnow()
    }
    
    if existing_rating:
        await db.practice_ratings.update_one(
            {"_id": existing_rating["_id"]},
            {"$set": rating_doc}
        )
        return {"message": "Rating updated successfully"}
    else:
        result = await db.practice_ratings.insert_one(rating_doc)
        return {
            "rating_id": str(result.inserted_id),
            "message": "Rating submitted successfully"
        }


@app.get("/api/traditional-practices/{practice_id}/ratings")
async def get_practice_ratings(
    practice_id: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get ratings and reviews for a practice"""
    ratings = await db.practice_ratings.find({
        "practice_id": practice_id
    }).sort("created_at", -1).limit(limit).to_list(limit)
    
    result = []
    for r in ratings:
        user = await db.users.find_one({"_id": ObjectId(r["user_id"])})
        result.append({
            "rating": r["rating"],
            "review": r.get("review"),
            "ease_of_implementation": r.get("ease_of_implementation"),
            "effectiveness": r.get("effectiveness"),
            "would_recommend": r.get("would_recommend", True),
            "user_name": user["name"] if user else "Anonymous",
            "user_location": user.get("village") if user else None,
            "created_at": r["created_at"].isoformat()
        })
    
    return result


@app.get("/api/traditional-practices/search")
async def search_traditional_practices(
    query: str,
    current_user: dict = Depends(get_current_user)
):
    """Search traditional practices by keyword"""
    search_query = {
        "$or": [
            {"title": {"$regex": query, "$options": "i"}},
            {"description": {"$regex": query, "$options": "i"}},
            {"region": {"$regex": query, "$options": "i"}},
            {"tribe_name": {"$regex": query, "$options": "i"}},
            {"best_for_crops": {"$regex": query, "$options": "i"}}
        ]
    }
    
    practices = await db.traditional_practices.find(search_query).limit(20).to_list(20)
    
    return [
        {
            "id": str(p["_id"]),
            "title": p["title"],
            "description": p["description"],
            "region": p["region"],
            "tribe_name": p.get("tribe_name"),
            "category": p["category"]
        } for p in practices
    ]


@app.get("/api/traditional-practices/statistics/summary")
async def get_traditional_practices_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get statistics about traditional practices"""
    
    # Total practices
    total_practices = await db.traditional_practices.count_documents({})
    
    # By category
    category_pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    category_stats = await db.traditional_practices.aggregate(category_pipeline).to_list(10)
    
    # By region
    region_pipeline = [
        {"$group": {"_id": "$region", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    region_stats = await db.traditional_practices.aggregate(region_pipeline).to_list(5)
    
    # Verified by elders
    verified_count = await db.traditional_practices.count_documents({
        "verified_by_elders": True
    })
    
    return {
        "total_practices": total_practices,
        "verified_by_elders": verified_count,
        "by_category": [
            {"category": c["_id"], "count": c["count"]} for c in category_stats
        ],
        "by_region": [
            {"region": r["_id"], "count": r["count"]} for r in region_stats
        ]
    }

async def seed_traditional_practices():
    """గిరిజన సమాజాల నుండి సంప్రదాయ వ్యవసాయ పద్ధతులను సీడ్ చేయండి"""
    
    # సృష్టికర్తగా ఒక స్పెషలిస్ట్ యూజర్‌ను పొందండి
    specialist_user = await db.users.find_one({"role": "specialist"})
    created_by_id = str(specialist_user["_id"]) if specialist_user else "000000000000000000000000"
    
    practices_data = [
        {
            "title": "చంద్ర దశల ప్రకారం విత్తనాలు",
            "description": "చంద్ర దశల ఆధారంగా విత్తనాలు నాటడం ద్వారా గరిష్ట వృద్ధి సాధించవచ్చు. పచ్చి ఆకుల కూరగాయలకు పూర్తి చంద్ర ఉత్తమం, మూర్ఖ చంద్రమాసానికి మూల పంటలకు. ఈ పురాతన పద్ధతి భూగర్భ జల చలనం మరియు మొక్కల నీటి వసూళ్లపై ప్రభావం చూపే గురుత్వాకర్షణ శక్తులతో అనుసంధానిస్తుంది.",
            "category": "planting",
            "region": "ఆంధ్రప్రదేశ్‌లోని గిరిజన సమాజాలు",
            "tribe_name": "కోయా గిరిజనులు",
            "local_language": "తెలుగు, koya",
            "best_for_crops": ["అన్నం", "కూరగాయలు", "మూల పంటలు", "పచ్చికూరలు"],
            "season": "అన్ని సీజన్లు",
            "implements_needed": ["చంద్ర క్యాలెండర్", "విత్తనాలు", "ప్రాథమిక వ్యవసాయ పరికరాలు"],
            "duration": "పొదుపు సీజన్ మొత్తం కొనసాగుతుంది",
            "scientific_basis": "చంద్ర గురుత్వాకర్షణ భూమి మట్టిలోని నీటి మోతాదును మరియు మొక్కల నీటి గ్రహణాన్ని ప్రభావితం చేస్తుంది. సరైన చంద్ర దశలలో నాటినప్పుడు విత్తనాల వృద్ధి రేట్లు పెరుగుతాయి.",
            "local_names": {
                "telugu": "చంద్ర దశల ప్రకారం విత్తనాలు",
                "hindi": "चंद्र कला के अनुसार बुवाई"
            },
            "difficulty_level": "సులభం",
            "verified_by_elders": True,
            "elder_name": "రామయ్య దొర్ల",
            "elder_contact": "గ్రామ పెద్ద - భద్రాచలం",
            "success_stories": [
                {
                    "farmer_name": "లక్ష్మి",
                    "location": "ఖమ్మం జిల్లా",
                    "result": "చంద్ర దశల ప్రకారం విత్తనాలు నాటడం వల్ల కూరగాయల ఉత్పత్తి 30% పెరిగింది"
                }
            ],
            "media_urls": [],
            "video_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "మూడు సహోదరుల పద్ధతి",
            "description": "కత్తి, పప్పు మరియు చెరకు పంటలను కలిపి నాటడం. కత్తి పప్పు కోసం మద్దతు ఇస్తుంది, పప్పు నేలలో నత్రజని స్థిరం చేస్తుంది, చెరకు ఆకులు మట్టిని కప్పి నీటి పరిమాణాన్ని నిలుపుతాయి మరియు పొలంలో ఇల్లాలు తడిసేలా నిరోధిస్తాయి. ఈ సంప్రదాయ పద్ధతి స్థలం మరియు నేల ఆరోగ్యాన్ని గరిష్టం చేస్తుంది.",
            "category": "planting",
            "region": "తూర్పు ఘట్టాల గిరిజన ప్రాంతాలు",
            "tribe_name": "సావర గిరిజనులు",
            "local_language": "సావర, తెలుగు",
            "best_for_crops": ["కత్తి", "పప్పు", "చెరకు", "పంప్కిన్"],
            "season": "ఖరీఫ్",
            "implements_needed": ["మూడు పంటల విత్తనాలు", "దూరం కొలిచే పరికరాలు"],
            "duration": "మొత్తం పంట సీజన్ (3-4 నెలలు)",
            "scientific_basis": "పప్పు ముల్లులలో ఉండే నత్రజని స్థిరం చేసే బాక్టీరియా నేలని సేంద్రియంగా న్యూట్రియెంట్ తో సంపూర్ణం చేస్తుంది. కత్తి దండు పప్పు కోసం సహజ ట్రెలిస్ అందిస్తుంది. చెరకు ఆకులు నీరు ఆవిరైజ్ కావడం తగ్గించటం మరియు నీటి నిల్వ చేస్తూ, కౌలు నిరోధిస్తుంది.",
            "local_names": {
                "telugu": "మూడు సహోదరుల పద్ధతి",
                "savara": "Tiini Peṇṭa Vidhi"
            },
            "difficulty_level": "మధ్యస్థం",
            "verified_by_elders": True,
            "elder_name": "మంగమ్మ సావర",
            "elder_contact": "గిరిజన నేత - విశాఖపట్నం ఏజెన్సీ",
            "success_stories": [
                {
                    "farmer_name": "శ్రీనివాస్",
                    "location": "అరాకు వ్యాలీ",
                    "result": "ఉత్పత్తులు తగ్గకుండా ఎరువుల వాడకం 40% తగ్గించబడింది"
                }
            ],
            "media_urls": [],
            "video_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "వేప విత్తన శుద్ధి",
            "description": "విత్తనాలను నాటే ముందు వేప పొడి మరియు గోరు మ manure తో కవర్ చేసి నేల వ్యాధులు మరియు pests నుండి రక్షించడం. వేప పొడి, గోరు మ manure, మట్టి సమానంగా మిక్స్ చేసి విత్తనాలను కవర్ చేసి, నీటినీడలో ఎండిన తరువాత నాటడం.",
            "category": "seed_treatment",
            "region": "గోదావరి గిరిజన ప్రాంతాలు",
            "tribe_name": "కొండా రెడ్డి గిరిజనులు",
            "local_language": "తెలుగు",
            "best_for_crops": ["అన్నం", "పత్తి", "పప్పు", "కూరగాయలు"],
            "season": "అన్ని సీజన్లు",
            "implements_needed": ["వేప పొడి", "గోరు మ manure", "మట్టి", "మిక్సింగ్ కంటెయినర్"],
            "duration": "నాటే ముందు 1-2 రోజుల ప్రిపరేషన్",
            "scientific_basis": "వేపలో అజడిరాక్టిన్ ఉంటుంది, ఇది pests కు వ్యతిరేకం మరియు insecticidal లక్షణాల కలిగినది. గోరు మ manure ఉపయోగకరమైన సూక్ష్మజీవులు కలిగి విత్తనాలను రక్షిస్తుంది. మట్టి కవర్ రక్షణ ఇస్తుంది.",
            "local_names": {
                "telugu": "వేప విత్తన శుద్ధి",
                "hindi": "नीम बीज उपचार"
            },
            "difficulty_level": "సులభం",
            "verified_by_elders": True,
            "elder_name": "సుబ్బారావు",
            "elder_contact": "వ్యవసాయ పెద్ద - తూర్పు గోదావరి",
            "success_stories": [
                {
                    "farmer_name": "రాజేష్",
                    "location": "రాజమండ్రి",
                    "result": "డ్యాంపింగ్ ఆఫ్ వ్యాధి వల్ల seedlings మృతి 80% తగ్గింది"
                }
            ],
            "media_urls": [],
            "video_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "బూడిద మరియు చార్కోల్ పురుగుల నియంత్రణ",
            "description": "మొక్కల చుట్టూ వుడ్ యాష్ మరియు చార్కోల్ పొడిని పసరించడం ద్వారా pests మరియు స్నైల్స్ ని నివారించడం. నేల pH మెరుగుపరచటం మరియు పొటాషియం అందించడం. మంచు ఉండే సమయంలో ఎరువుతో సస్పెన్షన్ కోసం వేళకి ఉపయోగించండి.",
            "category": "pest_control",
            "region": "అరణ్య అంచు గిరిజన ప్రాంతాలు",
            "tribe_name": "చెంచు గిరిజనులు",
            "local_language": "చెంచు, తెలుగు",
            "best_for_crops": ["కూరగాయలు", "మిల్లెట్స్", "అరటికాయ", "అల్లం"],
            "season": "అన్ని సీజన్లు",
            "implements_needed": ["వుడ్ యాష్", "చార్కోల్ పొడి", "పసరించే పరికరం"],
            "duration": "ప్రతి వారం లేదా వర్షం తర్వాత వర్తింపచేయండి",
            "scientific_basis": "వుడ్ యాష్ pests కి abrasive గా ఉంటుంది, అల్కలైన్ pH చాలా pests ని నిరోధిస్తుంది. చార్కోల్ నేల నిర్మాణం మరియు నీటి నిల్వ మెరుగుపరుస్తుంది. పొటాషియం మొక్కల కణాలను బలంగా చేస్తుంది, pests కి మరింత ప్రతిఘటన ఇస్తుంది.",
            "local_names": {
                "telugu": "బూడిద పురుగుల నివారణ",
                "chenchu": "Buddida Purugu Tolagiñchu"
            },
            "difficulty_level": "సులభం",
            "verified_by_elders": True,
            "elder_name": "రాములు చెంచు",
            "elder_contact": "అరణ్య పెద్ద - నల్లమల అరణ్య",
            "success_stories": [
                {
                    "farmer_name": "వెంకన్న",
                    "location": "అత్మకూరు",
                    "result": "కూరగాయలపై slug నష్టం పూర్తిగా తొలగించబడింది"
                }
            ],
            "media_urls": [],
            "video_urls": [],
            "created_by": created_by_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        # మీరు మిగతా practices కోసం కూడా ఇదే ఫార్మాట్ లోకి అనువదించవచ్చు
    ]
    
    try:
        existing_count = await db.traditional_practices.count_documents({})
        if existing_count > 0:
            print(f"సంప్రదాయ పద్ధతులు ఇప్పటికే ఉన్నాయి ({existing_count} రికార్డులు). సీడ్ దాటించబడింది.")
            return
        
        result = await db.traditional_practices.insert_many(practices_data)
        print(f"{len(result.inserted_ids)} సంప్రదాయ పద్ధతులను విజయవంతంగా సీడ్ చేయబడింది!")
        
    except Exception as e:
        print(f"సంప్రదాయ పద్ధతులను సీడ్ చేయడానికి లోపం: {e}")

@app.post("/api/admin/seed-traditional-practices")
async def trigger_seed_traditional_practices(
    current_user: dict = Depends(require_role("specialist"))
):
    """Seed traditional practices data - Specialist only"""
    await seed_traditional_practices()
    return {"message": "Traditional practices seeded successfully"}

# Seasonal Advisory System (SAS) Integration

# Add these endpoints to your main.py file

# ============= Seasonal Calendar & Advisory Routes =============

@app.get("/api/seasonal-calendar")
async def get_seasonal_calendar(
    crop_id: Optional[str] = None,
    month: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get seasonal calendar with monthly agricultural activities"""
    
    query = {}
    if crop_id:
        query["crop_id"] = crop_id
    if month:
        query["month"] = month
    
    calendar_entries = await db.advisory_calendar.find(query).to_list(100)
    
    if not calendar_entries:
        # Return default calendar if no entries exist
        default_calendar = [
            {
                "month": "January",
                "season": "Rabi",
                "temperature": "Cool (10-25°C)",
                "activities": [
                    "Harvest rabi crops (wheat, chickpea, mustard)",
                    "Prepare for summer crop sowing",
                    "Monitor for late blight in potatoes"
                ],
                "preventive_actions": [
                    "Apply organic mulch to retain moisture",
                    "Check irrigation systems"
                ],
                "recommended_crops": ["Wheat", "Chickpea", "Mustard", "Barley"]
            },
            {
                "month": "February",
                "season": "Rabi-Summer transition",
                "temperature": "Warm (15-30°C)",
                "activities": [
                    "Sowing of summer crops (vegetables, pulses)",
                    "Pest monitoring for aphids and whiteflies",
                    "Prepare soil for kharif season"
                ],
                "preventive_actions": [
                    "Install yellow sticky traps",
                    "Apply neem-based pesticides"
                ],
                "recommended_crops": ["Cucumber", "Watermelon", "Muskmelon", "Bottle Gourd"]
            },
            {
                "month": "March",
                "season": "Summer",
                "temperature": "Hot (20-35°C)",
                "activities": [
                    "Irrigation management critical",
                    "Apply organic fertilizers",
                    "Watch for heat stress in crops"
                ],
                "preventive_actions": [
                    "Mulching to conserve moisture",
                    "Drip irrigation setup",
                    "Shade nets for vegetables"
                ],
                "recommended_crops": ["Okra", "Bitter Gourd", "Ridge Gourd"]
            },
            {
                "month": "April",
                "season": "Summer",
                "temperature": "Very Hot (25-40°C)",
                "activities": [
                    "Pre-monsoon soil preparation",
                    "Deep ploughing",
                    "Collect and store farm yard manure"
                ],
                "preventive_actions": [
                    "Water conservation measures",
                    "Check bund maintenance"
                ],
                "recommended_crops": ["Green Gram", "Black Gram", "Sesame"]
            },
            {
                "month": "May",
                "season": "Summer",
                "temperature": "Very Hot (30-45°C)",
                "activities": [
                    "Summer crop maintenance",
                    "Water conservation critical",
                    "Prepare nursery for kharif crops"
                ],
                "preventive_actions": [
                    "Mulching with crop residues",
                    "Install rainwater harvesting"
                ],
                "recommended_crops": ["Maize", "Sorghum", "Pearl Millet"]
            },
            {
                "month": "June",
                "season": "Kharif (Monsoon onset)",
                "temperature": "Monsoon (25-35°C)",
                "activities": [
                    "Kharif sowing begins with monsoon arrival",
                    "Rice transplanting",
                    "Cotton sowing"
                ],
                "preventive_actions": [
                    "Ensure proper drainage",
                    "Monitor for stem borers",
                    "Apply organic growth promoters"
                ],
                "recommended_crops": ["Rice", "Cotton", "Soybean", "Groundnut"]
            },
            {
                "month": "July",
                "season": "Kharif",
                "temperature": "Monsoon (24-32°C)",
                "activities": [
                    "Weeding and intercultural operations",
                    "Pest and disease monitoring",
                    "Apply first top dressing"
                ],
                "preventive_actions": [
                    "Install pheromone traps",
                    "Monitor for leaf blast in rice",
                    "Drainage management"
                ],
                "recommended_crops": ["Finger Millet", "Pigeon Pea", "Green Gram"]
            },
            {
                "month": "August",
                "season": "Kharif",
                "temperature": "Monsoon (23-30°C)",
                "activities": [
                    "Peak vegetative growth stage",
                    "Second weeding",
                    "Pest scouting"
                ],
                "preventive_actions": [
                    "Apply bio-pesticides",
                    "Check for fungal diseases",
                    "Maintain field sanitation"
                ],
                "recommended_crops": ["Vegetables (Tomato, Brinjal, Chili)"]
            },
            {
                "month": "September",
                "season": "Kharif",
                "temperature": "Post-monsoon (22-30°C)",
                "activities": [
                    "Flowering and grain formation",
                    "Monitor for pests and diseases",
                    "Prepare for harvest"
                ],
                "preventive_actions": [
                    "Bird scaring for ripening crops",
                    "Check for lodging in cereals",
                    "Apply organic pesticides if needed"
                ],
                "recommended_crops": ["Late-sown vegetables"]
            },
            {
                "month": "October",
                "season": "Kharif-Rabi transition",
                "temperature": "Cool (18-28°C)",
                "activities": [
                    "Kharif harvest begins",
                    "Rabi field preparation",
                    "Soil testing"
                ],
                "preventive_actions": [
                    "Proper drying of harvested grain",
                    "Storage pest management",
                    "Apply FYM for rabi crops"
                ],
                "recommended_crops": ["Potato", "Onion", "Garlic"]
            },
            {
                "month": "November",
                "season": "Rabi",
                "temperature": "Cool (15-25°C)",
                "activities": [
                    "Rabi sowing (wheat, chickpea)",
                    "Irrigation scheduling",
                    "Apply basal fertilizers"
                ],
                "preventive_actions": [
                    "Seed treatment with organic agents",
                    "Monitor for termites",
                    "Proper spacing for crops"
                ],
                "recommended_crops": ["Wheat", "Chickpea", "Mustard", "Lentil"]
            },
            {
                "month": "December",
                "season": "Rabi",
                "temperature": "Cold (10-20°C)",
                "activities": [
                    "Early vegetative stage management",
                    "Weed control",
                    "Frost protection measures"
                ],
                "preventive_actions": [
                    "Light irrigation during dry spells",
                    "Monitor for aphids in mustard",
                    "Apply organic mulch if frost expected"
                ],
                "recommended_crops": ["Vegetables (Cauliflower, Cabbage, Peas)"]
            }
        ]
        
        return {
            "current_month": datetime.utcnow().strftime("%B"),
            "calendar": default_calendar
        }
    
    # Format existing entries
    calendar_data = []
    for entry in calendar_entries:
        crop = await db.crops.find_one({"_id": ObjectId(entry["crop_id"])}) if entry.get("crop_id") else None
        
        calendar_data.append({
            "id": str(entry["_id"]),
            "month": entry["month"],
            "crop_name": crop["name"] if crop else "General",
            "preventive_actions": entry.get("preventive_actions", []),
            "treatment_alerts": entry.get("treatment_alerts", []),
            "weather_alerts": entry.get("weather_alerts", [])
        })
    
    return {
        "current_month": datetime.utcnow().strftime("%B"),
        "calendar": calendar_data
    }


@app.get("/api/seasonal-calendar/current")
async def get_current_season_info(
    current_user: dict = Depends(get_current_user)
):
    """Get current season information and recommendations"""
    
    current_month = datetime.utcnow().month
    
    # Determine season based on month
    if current_month in [11, 12, 1, 2]:
        season = "Rabi"
        season_desc = "Winter cropping season"
        crops = ["Wheat", "Chickpea", "Mustard", "Barley", "Lentil"]
    elif current_month in [6, 7, 8, 9, 10]:
        season = "Kharif"
        season_desc = "Monsoon cropping season"
        crops = ["Rice", "Cotton", "Soybean", "Maize", "Groundnut"]
    else:
        season = "Summer/Zaid"
        season_desc = "Summer cropping season"
        crops = ["Watermelon", "Muskmelon", "Cucumber", "Vegetables"]
    
    # Get weather alerts for user location
    location = current_user.get("district", "")
    weather_alerts = await db.weather_alerts.find(
        {"location": location}
    ).sort("timestamp", -1).limit(5).to_list(5)
    
    return {
        "season": season,
        "description": season_desc,
        "recommended_crops": crops,
        "current_month": datetime.utcnow().strftime("%B"),
        "weather_alerts": [
            {
                "type": w["alert_type"],
                "message": w["message"],
                "action": w.get("recommended_action"),
                "timestamp": w["timestamp"].isoformat()
            } for w in weather_alerts
        ]
    }


@app.post("/api/seasonal-calendar/entry")
async def create_calendar_entry(
    entry_data: dict,
    current_user: dict = Depends(require_role("specialist"))
):
    """Create a seasonal calendar entry (specialists only)"""
    
    calendar_entry = {
        "crop_id": entry_data.get("crop_id"),
        "month": entry_data["month"],
        "preventive_actions": entry_data.get("preventive_actions", []),
        "treatment_alerts": entry_data.get("treatment_alerts", []),
        "weather_alerts": entry_data.get("weather_alerts", []),
        "created_by": str(current_user["_id"]),
        "created_at": datetime.utcnow()
    }
    
    result = await db.advisory_calendar.insert_one(calendar_entry)
    
    return {
        "entry_id": str(result.inserted_id),
        "message": "Calendar entry created successfully"
    }


@app.get("/api/seasonal-calendar/crop-recommendations")
async def get_crop_recommendations_by_season(
    season: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get crop recommendations based on season"""
    
    if not season:
        # Auto-detect current season
        current_month = datetime.utcnow().month
        if current_month in [11, 12, 1, 2]:
            season = "rabi"
        elif current_month in [6, 7, 8, 9, 10]:
            season = "kharif"
        else:
            season = "summer"
    
    # Hardcoded recommendations as fallback - THIS IS THE KEY ADDITION
    season_crops = {
        "rabi": [
            {"name": "Wheat", "local_name": "గోధుమ", "soil_types": ["Loamy", "Clay"], "disease_risk": "Low"},
            {"name": "Chickpea", "local_name": "శనగలు", "soil_types": ["Loamy", "Black"], "disease_risk": "Medium"},
            {"name": "Mustard", "local_name": "ఆవాలు", "soil_types": ["Loamy"], "disease_risk": "Low"},
            {"name": "Barley", "local_name": "బార్లీ", "soil_types": ["Loamy", "Sandy"], "disease_risk": "Low"},
            {"name": "Lentil", "local_name": "మసూర్", "soil_types": ["Loamy"], "disease_risk": "Low"},
            {"name": "Peas", "local_name": "బఠానీలు", "soil_types": ["Loamy", "Clay"], "disease_risk": "Low"},
        ],
        "kharif": [
            {"name": "Rice", "local_name": "వరి", "soil_types": ["Clay", "Loamy"], "disease_risk": "Medium"},
            {"name": "Cotton", "local_name": "పత్తి", "soil_types": ["Black", "Loamy"], "disease_risk": "High"},
            {"name": "Soybean", "local_name": "సోయాబీన్", "soil_types": ["Loamy", "Black"], "disease_risk": "Medium"},
            {"name": "Maize", "local_name": "మొక్కజొన్న", "soil_types": ["Loamy"], "disease_risk": "Medium"},
            {"name": "Groundnut", "local_name": "వేరుశనగ", "soil_types": ["Sandy", "Loamy"], "disease_risk": "Medium"},
            {"name": "Pigeon Pea", "local_name": "కందులు", "soil_types": ["Black", "Red"], "disease_risk": "Low"},
        ],
        "summer": [
            {"name": "Watermelon", "local_name": "పుచ్చకాయ", "soil_types": ["Sandy", "Loamy"], "disease_risk": "Low"},
            {"name": "Cucumber", "local_name": "దోసకాయ", "soil_types": ["Loamy"], "disease_risk": "Low"},
            {"name": "Muskmelon", "local_name": "ఖర్బూజా", "soil_types": ["Sandy", "Loamy"], "disease_risk": "Low"},
            {"name": "Bottle Gourd", "local_name": "సొరకాయ", "soil_types": ["Loamy"], "disease_risk": "Low"},
            {"name": "Okra", "local_name": "బెండకాయ", "soil_types": ["Loamy"], "disease_risk": "Medium"},
            {"name": "Bitter Gourd", "local_name": "కాకరకాయ", "soil_types": ["Loamy"], "disease_risk": "Low"},
        ]
    }
    
    # Start with hardcoded recommendations
    recommendations = season_crops.get(season.lower(), [])
    
    # Try to fetch from database and merge (optional)
    try:
        crops = await db.crops.find(
            {"season": {"$regex": season, "$options": "i"}}
        ).to_list(100)
        
        # Add database crops if any exist (avoid duplicates)
        existing_names = {rec["name"].lower() for rec in recommendations}
        
        for crop in crops:
            if crop["name"].lower() not in existing_names:
                disease_count = await db.crop_photos.count_documents({
                    "crop_id": str(crop["_id"]),
                    "disease": {"$ne": None}
                })
                
                recommendations.append({
                    "crop_id": str(crop["_id"]),
                    "name": crop["name"],
                    "local_name": crop.get("local_name"),
                    "soil_types": crop.get("soil_type", []),
                    "disease_risk": "High" if disease_count > 10 else "Medium" if disease_count > 5 else "Low"
                })
    except Exception as e:
        print(f"Error fetching crops from database: {e}")
        # Continue with hardcoded data
    
    print(f"Season: {season}, Recommendations count: {len(recommendations)}")
    
    return {
        "season": season,
        "recommendations": recommendations
    }

@app.get("/api/seasonal-calendar/activities/{month}")
async def get_monthly_activities(
    month: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed activities for a specific month"""
    
    # Get calendar entries for the month
    entries = await db.advisory_calendar.find(
        {"month": {"$regex": month, "$options": "i"}}
    ).to_list(100)
    
    if not entries:
        return {
            "month": month,
            "message": "No specific activities recorded for this month",
            "activities": []
        }
    
    activities = []
    for entry in entries:
        crop = await db.crops.find_one({"_id": ObjectId(entry["crop_id"])}) if entry.get("crop_id") else None
        
        activities.append({
            "crop": crop["name"] if crop else "General",
            "preventive_actions": entry.get("preventive_actions", []),
            "treatment_alerts": entry.get("treatment_alerts", []),
            "weather_considerations": entry.get("weather_alerts", [])
        })
    
    return {
        "month": month,
        "activities": activities
    }



# Add these models to your models.py file

class VideoTutorial(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description: str
    category: str  # pest_control, soil_management, composting, irrigation, etc.
    duration_minutes: int
    language: str = "telugu"
    video_url: str
    thumbnail_url: Optional[str] = None
    
    # Content details
    topics_covered: List[str] = []
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced
    
    # Targeting
    suitable_for_crops: List[str] = []
    season_relevance: List[str] = []
    
    # Engagement metrics
    views_count: int = 0
    likes_count: int = 0
    completion_rate: float = 0.0
    
    # Metadata
    created_by: str
    verified_by_specialist: bool = False
    is_featured: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class VideoProgress(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    video_id: str
    watched_duration_seconds: int = 0
    total_duration_seconds: int
    progress_percentage: float = 0.0
    completed: bool = False
    last_watched_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class VideoRating(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    video_id: str
    user_id: str
    rating: int  # 1-5
    review: Optional[str] = None
    helpful: Optional[bool] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request Models
class VideoCreateRequest(BaseModel):
    title: str
    description: str
    category: str
    duration_minutes: int
    language: str = "telugu"
    video_url: str
    thumbnail_url: Optional[str] = None
    topics_covered: List[str] = []
    difficulty_level: str = "beginner"
    suitable_for_crops: List[str] = []
    season_relevance: List[str] = []


class VideoUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    duration_minutes: Optional[int] = None
    topics_covered: Optional[List[str]] = None
    difficulty_level: Optional[str] = None
    suitable_for_crops: Optional[List[str]] = None
    season_relevance: Optional[List[str]] = None
    is_featured: Optional[bool] = None


class VideoProgressUpdateRequest(BaseModel):
    watched_duration_seconds: int
    total_duration_seconds: int


class VideoRatingRequest(BaseModel):
    rating: int
    review: Optional[str] = None
    helpful: Optional[bool] = None


# ============= Video Tutorials Routes =============
# Add these routes to your main.py file

# @app.post("/api/video-tutorials/upload")
# async def upload_video_tutorial(
#     file: UploadFile = File(...),
#     title: str = None,
#     description: str = None,
#     category: str = None,
#     duration_minutes: int = None,
#     language: str = "telugu",
#     current_user: dict = Depends(get_current_user)
# ):
#     """Upload video file to Cloudinary and create tutorial"""
#     if current_user["role"] not in ["specialist", "admin"]:
#         raise HTTPException(
#             status_code=403,
#             detail="Only specialists and admins can upload videos"
#         )

#     # Validate file type
#     if not file.content_type.startswith("video/"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Must be a video file")

#     # ✅ Save to a temporary file (safer for Cloudinary upload)
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#             contents = await file.read()
#             tmp.write(contents)
#             tmp_path = tmp.name

#         # ✅ Upload to Cloudinary as video
#         upload_result = cloudinary.uploader.upload(
#             tmp_path,
#             folder="video_tutorials",
#             resource_type="video",
#             eager=[{"format": "jpg", "width": 300, "height": 200, "crop": "thumb"}]  # generate thumbnail
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

#     video_doc = {
#         "title": title or f"Video Tutorial - {datetime.utcnow().strftime('%Y-%m-%d')}",
#         "description": description or "",
#         "category": category or "general",
#         "duration_minutes": duration_minutes or 0,
#         "language": language,
#         "video_url": upload_result["secure_url"],
#         "thumbnail_url": upload_result["eager"][0]["secure_url"] if "eager" in upload_result else None,
#         "topics_covered": [],
#         "difficulty_level": "beginner",
#         "suitable_for_crops": [],
#         "season_relevance": [],
#         "views_count": 0,
#         "likes_count": 0,
#         "completion_rate": 0.0,
#         "created_by": str(current_user["_id"]),
#         "verified_by_specialist": current_user["role"] == "specialist",
#         "is_featured": False,
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow()
#     }

#     result = await db.video_tutorials.insert_one(video_doc)

#     return {
#         "video_id": str(result.inserted_id),
#         "video_url": upload_result["secure_url"],
#         "thumbnail_url": video_doc["thumbnail_url"],
#         "message": "Video uploaded successfully"
#     }

@app.post("/api/video-tutorials")
async def create_video_tutorial(
    video_data: VideoCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new video tutorial (admin/specialist only)"""
    if current_user["role"] not in ["specialist", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Only specialists and admins can create videos"
        )
    
    video_doc = {
        "title": video_data.title,
        "description": video_data.description,
        "category": video_data.category,
        "duration_minutes": video_data.duration_minutes,
        "language": video_data.language,
        "video_url": video_data.video_url,
        "thumbnail_url": video_data.thumbnail_url,
        "topics_covered": video_data.topics_covered,
        "difficulty_level": video_data.difficulty_level,
        "suitable_for_crops": video_data.suitable_for_crops,
        "season_relevance": video_data.season_relevance,
        "views_count": 0,
        "likes_count": 0,
        "completion_rate": 0.0,
        "created_by": str(current_user["_id"]),
        "verified_by_specialist": current_user["role"] == "specialist",
        "is_featured": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.video_tutorials.insert_one(video_doc)
    
    return {
        "video_id": str(result.inserted_id),
        "message": "Video tutorial created successfully"
    }



@app.post("/api/video-tutorials/upload")
async def upload_video_tutorial(
    file: Optional[UploadFile] = File(None),
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    duration_minutes: int = Form(...),
    language: str = Form("telugu"),
    difficulty_level: str = Form("beginner"),
    youtube_url: Optional[str] = Form(None),  # NEW: YouTube URL support
    current_user: dict = Depends(get_current_user)
):
    """Upload video file to Cloudinary OR add YouTube video"""
    if current_user["role"] not in ["specialist", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Only specialists and admins can upload videos"
        )
    
    video_url = None
    thumbnail_url = None
    video_source = "uploaded"  # or "youtube"
    
    # Check if YouTube URL is provided
    if youtube_url:
        # Validate YouTube URL
        if not ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
            raise HTTPException(
                status_code=400,
                detail="Invalid YouTube URL"
            )
        
        video_url = youtube_url
        video_source = "youtube"
        
        # Extract YouTube video ID for thumbnail
        import re
        youtube_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if youtube_id_match:
            youtube_id = youtube_id_match.group(1)
            thumbnail_url = f"https://img.youtube.com/vi/{youtube_id}/maxresdefault.jpg"
        
        print(f"Adding YouTube video: {youtube_url}")
    
    elif file:
        # Original file upload logic
        if not file.content_type.startswith("video/"):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Must be a video file (mp4, avi, mov, etc.)"
            )
        
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > 100:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is 100MB"
            )
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            
            print(f"Uploading video: {file.filename} ({file_size_mb:.1f}MB)")
            
            upload_result = cloudinary.uploader.upload(
                tmp_path,
                folder="video_tutorials",
                resource_type="video",
                chunk_size=6000000,
                eager=[
                    {
                        "width": 640, 
                        "height": 360, 
                        "crop": "pad",
                        "format": "jpg",
                        "transformation": [{"start_offset": "0"}]
                    }
                ],
                eager_async=False
            )
            
            import os
            os.unlink(tmp_path)
            
            video_url = upload_result["secure_url"]
            
            if "eager" in upload_result and len(upload_result["eager"]) > 0:
                thumbnail_url = upload_result["eager"][0]["secure_url"]
            
            print(f"Video uploaded successfully: {video_url}")
            
        except Exception as e:
            print(f"Error uploading video: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload video: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Either video file or YouTube URL must be provided"
        )
    
    video_doc = {
        "title": title,
        "description": description,
        "category": category,
        "duration_minutes": duration_minutes,
        "language": language,
        "video_url": video_url,
        "video_source": video_source,  # NEW: Track video source
        "thumbnail_url": thumbnail_url,
        "topics_covered": [],
        "difficulty_level": difficulty_level,
        "suitable_for_crops": [],
        "season_relevance": [],
        "views_count": 0,
        "likes_count": 0,
        "completion_rate": 0.0,
        "created_by": str(current_user["_id"]),
        "verified_by_specialist": current_user["role"] == "specialist",
        "is_featured": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.video_tutorials.insert_one(video_doc)
    
    return {
        "video_id": str(result.inserted_id),
        "video_url": video_url,
        "video_source": video_source,
        "thumbnail_url": thumbnail_url,
        "message": "Video added successfully"
    }

# Place these BEFORE the {video_id} route
@app.get("/api/video-tutorials/my-progress")
async def get_my_video_progress(current_user: dict = Depends(get_current_user)):
    """Get user's video watching progress"""
    progress_list = await db.video_progress.find({
        "user_id": str(current_user["_id"])
    }).sort("last_watched_at", -1).to_list(100)
    
    result = []
    for progress in progress_list:
        video = await db.video_tutorials.find_one({"_id": ObjectId(progress["video_id"])})
        if video:
            result.append({
                "video_id": progress["video_id"],
                "video_title": video["title"],
                "progress_percentage": progress["progress_percentage"],
                "completed": progress["completed"],
                "last_watched_at": progress["last_watched_at"].isoformat()
            })
    
    return result


@app.get("/api/video-tutorials/categories")
async def get_video_categories(current_user: dict = Depends(get_current_user)):
    """Get list of video categories with counts"""
    pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    categories = await db.video_tutorials.aggregate(pipeline).to_list(100)
    
    return [
        {"category": c["_id"], "count": c["count"]}
        for c in categories
    ]
@app.get("/api/video-tutorials")
async def get_video_tutorials(
    category: Optional[str] = None,
    language: Optional[str] = None,
    difficulty: Optional[str] = None,
    crop: Optional[str] = None,
    season: Optional[str] = None,
    featured_only: bool = False,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get list of video tutorials with filters"""
    query = {}
    
    if category:
        query["category"] = category
    
    if language:
        query["language"] = language
    
    if difficulty:
        query["difficulty_level"] = difficulty
    
    if crop:
        query["suitable_for_crops"] = {"$in": [crop]}
    
    if season:
        query["season_relevance"] = {"$in": [season]}
    
    if featured_only:
        query["is_featured"] = True
    
    videos = await db.video_tutorials.find(query).sort(
        "created_at", -1
    ).limit(limit).to_list(limit)
    
    result = []
    for video in videos:
        progress = await db.video_progress.find_one({
            "user_id": str(current_user["_id"]),
            "video_id": str(video["_id"])
        })
        
        result.append({
            "id": str(video["_id"]),
            "title": video["title"],
            "description": video["description"],
            "category": video["category"],
            "duration": f"{video['duration_minutes']} mins",
            "language": video["language"],
            "video_url": video["video_url"],
            "video_source": video.get("video_source", "uploaded"),  # NEW
            "thumbnail_url": video.get("thumbnail_url"),
            "difficulty_level": video.get("difficulty_level", "beginner"),
            "topics_covered": video.get("topics_covered", []),
            "views_count": video.get("views_count", 0),
            "likes_count": video.get("likes_count", 0),
            "is_featured": video.get("is_featured", False),
            "user_progress": progress["progress_percentage"] if progress else 0,
            "completed": progress["completed"] if progress else False,
            "created_at": video["created_at"].isoformat()
        })
    
    return result
@app.get("/api/video-tutorials/{video_id}")
async def get_video_detail(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific video"""
    video = await db.video_tutorials.find_one({"_id": ObjectId(video_id)})
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get average rating
    ratings_pipeline = [
        {"$match": {"video_id": video_id}},
        {"$group": {
            "_id": None,
            "avg_rating": {"$avg": "$rating"},
            "total_ratings": {"$sum": 1}
        }}
    ]
    rating_result = await db.video_ratings.aggregate(ratings_pipeline).to_list(1)
    
    avg_rating = rating_result[0]["avg_rating"] if rating_result else 0
    total_ratings = rating_result[0]["total_ratings"] if rating_result else 0
    
    # Increment view count
    await db.video_tutorials.update_one(
        {"_id": ObjectId(video_id)},
        {"$inc": {"views_count": 1}}
    )
    
    return {
        "id": str(video["_id"]),
        "title": video["title"],
        "description": video["description"],
        "category": video["category"],
        "duration_minutes": video["duration_minutes"],
        "language": video["language"],
        "video_url": video["video_url"],
        "video_source": video.get("video_source", "uploaded"),  # NEW
        "thumbnail_url": video.get("thumbnail_url"),
        "difficulty_level": video.get("difficulty_level", "beginner"),
        "topics_covered": video.get("topics_covered", []),
        "suitable_for_crops": video.get("suitable_for_crops", []),
        "season_relevance": video.get("season_relevance", []),
        "views_count": video.get("views_count", 0),
        "likes_count": video.get("likes_count", 0),
        "average_rating": round(avg_rating, 1),
        "total_ratings": total_ratings,
        "is_featured": video.get("is_featured", False),
        "created_at": video["created_at"].isoformat()
    }
@app.put("/api/video-tutorials/{video_id}")
async def update_video_tutorial(
    video_id: str,
    update_data: VideoUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update video tutorial"""
    video = await db.video_tutorials.find_one({"_id": ObjectId(video_id)})
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video["created_by"] != str(current_user["_id"]) and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to update this video")
    
    update_fields = update_data.dict(exclude_unset=True)
    update_fields["updated_at"] = datetime.utcnow()
    
    await db.video_tutorials.update_one(
        {"_id": ObjectId(video_id)},
        {"$set": update_fields}
    )
    
    return {"message": "Video updated successfully"}


@app.delete("/api/video-tutorials/{video_id}")
async def delete_video_tutorial(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete video tutorial"""
    video = await db.video_tutorials.find_one({"_id": ObjectId(video_id)})
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video["created_by"] != str(current_user["_id"]) and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this video")
    
    await db.video_tutorials.delete_one({"_id": ObjectId(video_id)})
    
    return {"message": "Video deleted successfully"}




@app.post("/api/video-tutorials/{video_id}/like")
async def like_video(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Like a video tutorial"""
    video = await db.video_tutorials.find_one({"_id": ObjectId(video_id)})
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    await db.video_tutorials.update_one(
        {"_id": ObjectId(video_id)},
        {"$inc": {"likes_count": 1}}
    )
    
    return {"message": "Video liked successfully"}


@app.post("/api/video-tutorials/{video_id}/rate")
async def rate_video(
    video_id: str,
    rating_data: VideoRatingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Rate and review a video"""
    video = await db.video_tutorials.find_one({"_id": ObjectId(video_id)})
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    existing_rating = await db.video_ratings.find_one({
        "video_id": video_id,
        "user_id": str(current_user["_id"])
    })
    
    rating_doc = {
        "video_id": video_id,
        "user_id": str(current_user["_id"]),
        "rating": rating_data.rating,
        "review": rating_data.review,
        "helpful": rating_data.helpful,
        "created_at": datetime.utcnow()
    }
    
    if existing_rating:
        await db.video_ratings.update_one(
            {"_id": existing_rating["_id"]},
            {"$set": rating_doc}
        )
        return {"message": "Rating updated successfully"}
    else:
        result = await db.video_ratings.insert_one(rating_doc)
        return {
            "rating_id": str(result.inserted_id),
            "message": "Rating submitted successfully"
        }


@app.get("/api/video-tutorials/{video_id}/ratings")
async def get_video_ratings(
    video_id: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get ratings and reviews for a video"""
    ratings = await db.video_ratings.find({
        "video_id": video_id
    }).sort("created_at", -1).limit(limit).to_list(limit)
    
    result = []
    for r in ratings:
        user = await db.users.find_one({"_id": ObjectId(r["user_id"])})
        result.append({
            "rating": r["rating"],
            "review": r.get("review"),
            "helpful": r.get("helpful"),
            "user_name": user["name"] if user else "Anonymous",
            "created_at": r["created_at"].isoformat()
        })
    
    return result

@app.get("/api/video-tutorials/my-progress")
async def get_my_video_progress(current_user: dict = Depends(get_current_user)):
    """Get user's video watching progress"""
    progress_list = await db.video_progress.find({
        "user_id": str(current_user["_id"])
    }).sort("last_watched_at", -1).to_list(100)
    
    result = []
    for progress in progress_list:
        video = await db.video_tutorials.find_one({"_id": ObjectId(progress["video_id"])})
        if video:
            result.append({
                "video_id": progress["video_id"],
                "video_title": video["title"],
                "progress_percentage": progress["progress_percentage"],
                "completed": progress["completed"],
                "last_watched_at": progress["last_watched_at"].isoformat()
            })
    
    return result

@app.get("/api/debug/video-progress")
async def debug_video_progress(current_user: dict = Depends(get_current_user)):
    """Debug endpoint to check video progress"""
    user_id = str(current_user["_id"])
    
    # Get all video progress records
    all_progress = await db.video_progress.find({
        "user_id": user_id
    }).to_list(100)
    
    completed_count = await db.video_progress.count_documents({
        "user_id": user_id,
        "completed": True
    })
    
    return {
        "user_id": user_id,
        "total_videos_started": len(all_progress),
        "completed_videos": completed_count,
        "progress_records": [
            {
                "video_id": p["video_id"],
                "progress": p["progress_percentage"],
                "completed": p.get("completed", False),
                "watched_duration": p.get("watched_duration_seconds", 0),
                "total_duration": p.get("total_duration_seconds", 0)
            } for p in all_progress
        ]
    }

@app.put("/api/video-tutorials/{video_id}/progress")
async def update_video_progress(
    video_id: str,
    progress_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update video watching progress"""
    
    user_id = str(current_user["_id"])
    watched_seconds = progress_data.get("watched_duration_seconds", 0)
    total_seconds = progress_data.get("total_duration_seconds", 0)
    
    if total_seconds == 0:
        raise HTTPException(status_code=400, detail="Total duration cannot be 0")
    
    progress_percentage = (watched_seconds / total_seconds) * 100
    completed = progress_percentage >= 90  # Consider 90%+ as completed
    
    existing = await db.video_progress.find_one({
        "user_id": user_id,
        "video_id": video_id
    })
    
    progress_doc = {
        "user_id": user_id,
        "video_id": video_id,
        "watched_duration_seconds": watched_seconds,
        "total_duration_seconds": total_seconds,
        "progress_percentage": progress_percentage,
        "completed": completed,
        "last_watched_at": datetime.utcnow()
    }
    
    if completed and (not existing or not existing.get("completed")):
        progress_doc["completed_at"] = datetime.utcnow()
        
        # 🔥 KEY FIX: Update user_progress collection
        await db.user_progress.update_one(
            {"user_id": user_id},
            {
                "$inc": {"learning_sessions_completed": 1},
                "$set": {"last_updated": datetime.utcnow()}
            },
            upsert=True
        )
    
    if existing:
        await db.video_progress.update_one(
            {"_id": existing["_id"]},
            {"$set": progress_doc}
        )
    else:
        await db.video_progress.insert_one(progress_doc)
    
    return {
        "message": "Progress updated",
        "progress_percentage": round(progress_percentage, 1),
        "completed": completed
    }


#  vapi ai integration 
# ============= VAPI Voice Assistant Integration =============

@app.get("/api/vapi/config")
async def get_vapi_config():
    """Return VAPI configuration for frontend"""
    return {
        "publicKey": settings.VAPI_PUBLIC_KEY,
        "assistantId": settings.VAPI_ASSISTANT_ID
    }

@app.post("/api/vapi/query")
async def handle_vapi_query(
    query_data: VAPIQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Handle voice queries from VAPI assistant
    Process natural language queries and return structured responses
    """
    
    query = query_data.query.lower()
    user_id = str(current_user["_id"])
    response_text = ""
    suggested_actions = []
    relevant_data = None
    
    # Disease/Problem Queries
    if any(word in query for word in ["disease", "problem", "sick", "yellow", "spots", "dying"]):
        # Get recent crop photos
        recent_photos = await db.crop_photos.find(
            {"user_id": user_id}
        ).sort("uploaded_at", -1).limit(3).to_list(3)
        
        if recent_photos:
            latest = recent_photos[0]
            response_text = f"I see you recently uploaded a crop photo. It detected {latest.get('disease', 'unknown issue')} with {int(latest.get('confidence_score', 0)*100)}% confidence. "
            
            if latest.get('suggested_treatment'):
                response_text += "I have an organic treatment ready. Would you like me to explain it?"
            
            relevant_data = {
                "type": "disease_analysis",
                "photo_id": str(latest["_id"]),
                "disease": latest.get("disease"),
                "confidence": latest.get("confidence_score")
            }
            
            suggested_actions.append({
                "action": "view_analysis",
                "label": "View Full Analysis",
                "route": f"/crop-analysis/{latest['_id']}"
            })
        else:
            response_text = "To diagnose crop problems, please upload a clear photo of the affected plant. Go to Crop Analysis section and tap Upload Photo."
            suggested_actions.append({
                "action": "upload_photo",
                "label": "Upload Crop Photo",
                "route": "/crop-analysis"
            })
    
    # Organic Solutions Queries
    elif any(word in query for word in ["treatment", "cure", "organic", "pesticide", "spray"]):
        # Extract disease name if mentioned
        disease_keywords = query.split()
        
        solutions = await db.organic_solutions.find(
            {"diseases_treated": {"$exists": True}}
        ).sort("success_rate", -1).limit(3).to_list(3)
        
        if solutions:
            response_text = f"I found {len(solutions)} organic solutions. The top one is {solutions[0]['title']} with {solutions[0]['success_rate']}% success rate. "
            response_text += f"It costs around {solutions[0]['cost_per_acre']} rupees per acre. Should I share the recipe?"
            
            relevant_data = {
                "type": "organic_solutions",
                "solutions": [
                    {
                        "id": str(s["_id"]),
                        "title": s["title"],
                        "success_rate": s["success_rate"]
                    } for s in solutions
                ]
            }
            
            suggested_actions.append({
                "action": "view_solution",
                "label": "View Solution Details",
                "route": f"/organic-solutions/{solutions[0]['_id']}"
            })
        else:
            response_text = "Let me search for organic treatments. Can you tell me which disease or pest you're dealing with?"
    
    # Seasonal/Calendar Queries
    elif any(word in query for word in ["plant", "sow", "season", "month", "crop", "grow"]):
        current_month = datetime.utcnow().month
        
        # Determine season
        if current_month in [11, 12, 1, 2]:
            season = "Rabi"
            crops = ["Wheat", "Chickpea", "Mustard", "Barley"]
        elif current_month in [6, 7, 8, 9, 10]:
            season = "Kharif"
            crops = ["Rice", "Cotton", "Soybean", "Maize"]
        else:
            season = "Summer"
            crops = ["Watermelon", "Vegetables", "Cucumber"]
        
        response_text = f"Right now it's {season} season. Best crops to plant are {', '.join(crops[:3])}. "
        response_text += f"Would you like detailed month-wise activities?"
        
        relevant_data = {
            "type": "seasonal_info",
            "season": season,
            "crops": crops,
            "month": datetime.utcnow().strftime("%B")
        }
        
        suggested_actions.append({
            "action": "view_calendar",
            "label": "View Seasonal Calendar",
            "route": "/seasonal-calendar"
        })
    
    # Traditional Practices Queries
    elif any(word in query for word in ["traditional", "tribal", "old", "ancestors", "natural"]):
        practices = await db.traditional_practices.find(
            {"verified_by_elders": True}
        ).sort("created_at", -1).limit(3).to_list(3)
        
        if practices:
            response_text = f"I found {len(practices)} traditional practices. One is {practices[0]['title']} from {practices[0].get('region', 'various regions')}. "
            response_text += "It's been verified by tribal elders. Want to learn more?"
            
            relevant_data = {
                "type": "traditional_practices",
                "practices": [
                    {
                        "id": str(p["_id"]),
                        "title": p["title"],
                        "region": p.get("region")
                    } for p in practices
                ]
            }
            
            suggested_actions.append({
                "action": "view_practice",
                "label": "View Practice Details",
                "route": f"/traditional-practices/{practices[0]['_id']}"
            })
        else:
            response_text = "Traditional farming practices are available in the Traditional Knowledge section. Would you like me to guide you there?"
    
    # Weather Queries
    elif any(word in query for word in ["weather", "rain", "temperature", "forecast"]):
        location = current_user.get("district", "")
        
        weather_alerts = await db.weather_alerts.find(
            {"location": location}
        ).sort("timestamp", -1).limit(3).to_list(3)
        
        if weather_alerts:
            response_text = f"There are {len(weather_alerts)} weather alerts for {location}. "
            response_text += f"Latest: {weather_alerts[0]['message']} "
            response_text += f"Action needed: {weather_alerts[0].get('recommended_action', 'Monitor conditions')}"
            
            relevant_data = {
                "type": "weather_alerts",
                "alerts": [
                    {
                        "type": a["alert_type"],
                        "message": a["message"]
                    } for a in weather_alerts
                ]
            }
        else:
            response_text = f"No weather alerts for {location} right now. Conditions seem normal for farming."
    
    # Community Queries
    elif any(word in query for word in ["community", "forum", "farmers", "ask", "help"]):
        recent_posts = await db.community_posts.find(
            {"is_question": True, "is_solved": False}
        ).sort("created_at", -1).limit(5).to_list(5)
        
        response_text = f"There are {len(recent_posts)} active discussions in the community. "
        response_text += "Farmers are helping each other with crop problems. Would you like to join or post your question?"
        
        suggested_actions.append({
            "action": "view_community",
            "label": "View Community Forum",
            "route": "/community"
        })
    
    # Progress/Stats Queries
    elif any(word in query for word in ["progress", "stats", "crops monitored", "success"]):
        progress = await db.user_progress.find_one({"user_id": user_id})
        
        if progress:
            response_text = f"Great question! You've monitored {progress['crops_monitored']} crops and applied {progress['treatments_applied']} treatments. "
            response_text += f"Your success rate is {progress['success_rate']}%. Keep up the good work!"
            
            relevant_data = {
                "type": "user_progress",
                "stats": {
                    "crops_monitored": progress["crops_monitored"],
                    "treatments_applied": progress["treatments_applied"],
                    "success_rate": progress["success_rate"]
                }
            }
        else:
            response_text = "Let's start your farming journey! Upload your first crop photo to begin tracking progress."
    
    # Video Tutorial Queries
    elif any(word in query for word in ["video", "learn", "tutorial", "how to", "watch"]):
        videos = await db.video_tutorials.find(
            {"language": current_user.get("language_preference", "telugu")}
        ).sort("views_count", -1).limit(3).to_list(3)
        
        if videos:
            response_text = f"I have {len(videos)} helpful videos. Most popular is '{videos[0]['title']}' about {videos[0]['category']}. "
            response_text += f"It's {videos[0]['duration_minutes']} minutes long. Want to watch it?"
            
            suggested_actions.append({
                "action": "watch_video",
                "label": "Watch Tutorial",
                "route": f"/video-tutorials/{videos[0]['_id']}"
            })
        else:
            response_text = "Video tutorials are available in the Learning section. Shall I take you there?"
    
    # Default/Fallback
    else:
        response_text = "I'm here to help with crop diseases, organic treatments, seasonal advice, traditional practices, and farming tips. What would you like to know?"
        
        suggested_actions.extend([
            {"action": "upload", "label": "Analyze Crop Photo", "route": "/crop-analysis"},
            {"action": "solutions", "label": "Browse Organic Solutions", "route": "/organic-solutions"},
            {"action": "calendar", "label": "Seasonal Calendar", "route": "/seasonal-calendar"},
            {"action": "community", "label": "Community Forum", "route": "/community"}
        ])
    
    return VAPIResponse(
        response=response_text,
        suggested_actions=suggested_actions,
        relevant_data=relevant_data
    )


@app.get("/api/vapi/context")
async def get_vapi_context(current_user: dict = Depends(get_current_user)):
    """
    Provide context data for VAPI assistant
    """
    user_id = str(current_user["_id"])
    
    # Get user's recent activity
    recent_photos = await db.crop_photos.find(
        {"user_id": user_id}
    ).sort("uploaded_at", -1).limit(5).to_list(5)
    
    progress = await db.user_progress.find_one({"user_id": user_id})
    
    return {
        "user": {
            "name": current_user["name"],
            "location": current_user.get("district"),
            "language": current_user.get("language_preference", "telugu")
        },
        "recent_activity": {
            "photos_count": len(recent_photos),
            "last_upload": recent_photos[0]["uploaded_at"].isoformat() if recent_photos else None,
            "recent_diseases": [p.get("disease") for p in recent_photos if p.get("disease")]
        },
        "progress": {
            "crops_monitored": progress["crops_monitored"] if progress else 0,
            "treatments_applied": progress["treatments_applied"] if progress else 0,
            "success_rate": progress["success_rate"] if progress else 0
        },
        "current_season": "Rabi" if datetime.utcnow().month in [11,12,1,2] else "Kharif" if datetime.utcnow().month in [6,7,8,9,10] else "Summer"
    }


@app.post("/api/vapi/function-call")
async def handle_vapi_function_call(request: VAPIRequest):
    """
    Handle function calls from VAPI assistant
    VAPI will call this endpoint when it needs real-time data
    """
    
    function_name = request.message.functionCall.name if request.message.functionCall else None
    parameters = request.message.functionCall.parameters if request.message.functionCall else {}
    
    result = {}
    
    try:
        if function_name == "get_recent_disease_analysis":
            user_id = parameters.get("user_id")
            if user_id:
                photos = await db.crop_photos.find(
                    {"user_id": user_id}
                ).sort("uploaded_at", -1).limit(3).to_list(3)
                
                result = {
                    "success": True,
                    "data": [
                        {
                            "photo_id": str(p["_id"]),
                            "disease": p.get("disease", "Unknown"),
                            "confidence": round(p.get("confidence_score", 0) * 100, 1),
                            "crop": p.get("crop_id", ""),
                            "date": p["uploaded_at"].strftime("%Y-%m-%d")
                        } for p in photos
                    ]
                }
        
        elif function_name == "search_organic_solutions":
            disease = parameters.get("disease", "")
            solutions = await db.organic_solutions.find(
                {"diseases_treated": {"$regex": disease, "$options": "i"}}
            ).sort("success_rate", -1).limit(3).to_list(3)
            
            result = {
                "success": True,
                "data": [
                    {
                        "id": str(s["_id"]),
                        "title": s["title"],
                        "success_rate": s["success_rate"],
                        "cost_per_acre": s["cost_per_acre"],
                        "preparation_time": s["preparation_time"],
                        "ingredients": [ing["name"] for ing in s["ingredients"][:3]],
                        "application_method": s["application_method"]
                    } for s in solutions
                ]
            }
        
        elif function_name == "get_seasonal_calendar":
            current_month = datetime.utcnow().month
            month_name = datetime.utcnow().strftime("%B")
            
            if current_month in [11, 12, 1, 2]:
                season = "Rabi"
                crops = ["Wheat", "Chickpea", "Mustard", "Barley", "Lentil"]
                activities = [
                    "Harvest rabi crops",
                    "Irrigation management critical",
                    "Monitor for aphids and termites"
                ]
            elif current_month in [6, 7, 8, 9, 10]:
                season = "Kharif"
                crops = ["Rice", "Cotton", "Soybean", "Maize", "Groundnut"]
                activities = [
                    "Monsoon sowing begins",
                    "Rice transplanting",
                    "Pest and disease monitoring"
                ]
            else:
                season = "Summer"
                crops = ["Watermelon", "Muskmelon", "Cucumber", "Vegetables"]
                activities = [
                    "Water conservation critical",
                    "Mulching recommended",
                    "Heat stress management"
                ]
            
            result = {
                "success": True,
                "data": {
                    "current_month": month_name,
                    "season": season,
                    "recommended_crops": crops,
                    "key_activities": activities
                }
            }
        
        elif function_name == "get_user_progress":
            user_id = parameters.get("user_id")
            if user_id:
                progress = await db.user_progress.find_one({"user_id": user_id})
                user = await db.users.find_one({"_id": ObjectId(user_id)})
                
                if progress and user:
                    result = {
                        "success": True,
                        "data": {
                            "name": user["name"],
                            "crops_monitored": progress["crops_monitored"],
                            "treatments_applied": progress["treatments_applied"],
                            "success_rate": round(progress["success_rate"], 1),
                            "learning_sessions": progress["learning_sessions_completed"]
                        }
                    }
        
        elif function_name == "get_weather_alerts":
            location = parameters.get("location", "")
            alerts = await db.weather_alerts.find(
                {"location": {"$regex": location, "$options": "i"}}
            ).sort("timestamp", -1).limit(3).to_list(3)
            
            result = {
                "success": True,
                "data": [
                    {
                        "type": a["alert_type"],
                        "message": a["message"],
                        "recommended_action": a.get("recommended_action", "Monitor conditions"),
                        "date": a["timestamp"].strftime("%Y-%m-%d")
                    } for a in alerts
                ]
            }
        
        elif function_name == "get_traditional_practices":
            category = parameters.get("category", "")
            practices = await db.traditional_practices.find(
                {"category": {"$regex": category, "$options": "i"}} if category else {}
            ).sort("created_at", -1).limit(3).to_list(3)
            
            result = {
                "success": True,
                "data": [
                    {
                        "id": str(p["_id"]),
                        "title": p["title"],
                        "region": p.get("region", ""),
                        "tribe": p.get("tribe_name", ""),
                        "difficulty": p.get("difficulty_level", "medium"),
                        "verified": p.get("verified_by_elders", False)
                    } for p in practices
                ]
            }
        
        elif function_name == "get_community_discussions":
            limit = parameters.get("limit", 5)
            posts = await db.community_posts.find(
                {"is_question": True, "is_solved": False}
            ).sort("created_at", -1).limit(limit).to_list(limit)
            
            result = {
                "success": True,
                "data": [
                    {
                        "id": str(p["_id"]),
                        "title": p["title"],
                        "author": p.get("author_name", "Farmer"),
                        "tags": p.get("tags", []),
                        "comments_count": len(p.get("comments", []))
                    } for p in posts
                ]
            }
        
        elif function_name == "get_video_tutorials":
            category = parameters.get("category", "")
            language = parameters.get("language", "telugu")
            
            query = {"language": language}
            if category:
                query["category"] = {"$regex": category, "$options": "i"}
            
            videos = await db.video_tutorials.find(query).sort(
                "views_count", -1
            ).limit(3).to_list(3)
            
            result = {
                "success": True,
                "data": [
                    {
                        "id": str(v["_id"]),
                        "title": v["title"],
                        "category": v["category"],
                        "duration": f"{v['duration_minutes']} minutes",
                        "difficulty": v.get("difficulty_level", "beginner")
                    } for v in videos
                ]
            }
        
        elif function_name == "analyze_crop_photo":
            photo_id = parameters.get("photo_id")
            if photo_id:
                photo = await db.crop_photos.find_one({"_id": ObjectId(photo_id)})
                if photo:
                    result = {
                        "success": True,
                        "data": {
                            "disease": photo.get("disease", "Unknown"),
                            "confidence": round(photo.get("confidence_score", 0) * 100, 1),
                            "severity": photo.get("severity", "medium"),
                            "treatment_summary": photo.get("suggested_treatment", "")[:200]
                        }
                    }
        
        else:
            result = {
                "success": False,
                "error": f"Unknown function: {function_name}"
            }
    
    except Exception as e:
        result = {
            "success": False,
            "error": str(e)
        }
    
    return {"result": result}




@app.get("/api/vapi/user-context")
async def get_user_context_for_vapi(current_user: dict = Depends(get_current_user)):
    """Get user context to pass to VAPI"""
    user_id = str(current_user["_id"])
    
    # Get recent activity
    recent_photos = await db.crop_photos.find(
        {"user_id": user_id}
    ).sort("uploaded_at", -1).limit(5).to_list(5)
    
    progress = await db.user_progress.find_one({"user_id": user_id})
    
    return {
        "userId": user_id,
        "userName": current_user["name"],
        "location": current_user.get("district", ""),
        "language": current_user.get("language_preference", "telugu"),
        "recentActivity": {
            "totalPhotos": len(recent_photos),
            "lastUpload": recent_photos[0]["uploaded_at"].isoformat() if recent_photos else None,
            "recentDiseases": list(set([p.get("disease") for p in recent_photos if p.get("disease")]))
        },
        "progress": {
            "cropsMonitored": progress["crops_monitored"] if progress else 0,
            "treatmentsApplied": progress["treatments_applied"] if progress else 0,
            "successRate": progress["success_rate"] if progress else 0
        }
    }

#  whether related 
# ============= Configuration =============

# ============= Pydantic Models =============

class WeatherPreferencesUpdateRequest(BaseModel):
    enable_weather_alerts: Optional[bool] = None
    enable_rainfall_alerts: Optional[bool] = None
    enable_temperature_alerts: Optional[bool] = None
    enable_storm_alerts: Optional[bool] = None
    high_temp_threshold: Optional[float] = None
    low_temp_threshold: Optional[float] = None
    heavy_rain_threshold: Optional[float] = None
    morning_alert: Optional[bool] = None
    evening_alert: Optional[bool] = None
    primary_location: Optional[str] = None
    additional_locations: Optional[List[str]] = None


# ============= Configuration =============

WEATHER_API_KEY = "b43dbe4dd01521717c5d047fd3cb2b14"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5"


# ============= Weather Helper Functions =============

async def fetch_openweather_current(location: str):
    """Fetch current weather from OpenWeatherMap API"""
    try:
        url = f"{WEATHER_API_URL}/weather"
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None


async def fetch_openweather_forecast(location: str):
    """Fetch 5-day forecast from OpenWeatherMap API"""
    try:
        url = f"{WEATHER_API_URL}/forecast"
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "cnt": 40
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return None


async def generate_weather_alerts(location: str, weather_data: dict, preferences: dict):
    """Generate alerts based on current weather conditions and user preferences"""
    alerts = []
    
    if not weather_data:
        return alerts
    
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"] * 3.6  # Convert to km/h
    conditions = weather_data["weather"][0]["main"]
    rainfall = weather_data.get("rain", {}).get("1h", 0)
    
    # High temperature alert
    high_temp_threshold = preferences.get("high_temp_threshold", 35.0)
    if preferences.get("enable_temperature_alerts", True) and temp > high_temp_threshold:
        alert_key = f"high_temp_{location}_{datetime.utcnow().date()}"
        
        # Check if alert already exists today
        existing = await db.weather_alerts.find_one({
            "alert_key": alert_key,
            "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        if not existing:
            alert_doc = {
                "alert_key": alert_key,
                "location": location,
                "alert_type": "temperature",
                "message": f"High temperature alert: {round(temp)}°C. Heat stress may affect crops.",
                "recommended_action": "Increase irrigation frequency, provide shade for sensitive crops, and water during early morning or evening.",
                "severity": "high" if temp > high_temp_threshold + 5 else "medium",
                "affected_crops": ["Rice", "Tomato", "Vegetables"],
                "valid_until": datetime.utcnow() + timedelta(hours=12),
                "acknowledged_by": [],
                "timestamp": datetime.utcnow()
            }
            result = await db.weather_alerts.insert_one(alert_doc)
            alerts.append(str(result.inserted_id))
    
    # Low temperature alert
    low_temp_threshold = preferences.get("low_temp_threshold", 15.0)
    if preferences.get("enable_temperature_alerts", True) and temp < low_temp_threshold:
        alert_key = f"low_temp_{location}_{datetime.utcnow().date()}"
        
        existing = await db.weather_alerts.find_one({
            "alert_key": alert_key,
            "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        if not existing:
            alert_doc = {
                "alert_key": alert_key,
                "location": location,
                "alert_type": "temperature",
                "message": f"Low temperature alert: {round(temp)}°C. Protect frost-sensitive crops.",
                "recommended_action": "Use plastic covers for seedlings, protect sensitive crops, and delay irrigation if frost is expected.",
                "severity": "medium" if temp > 10 else "high",
                "affected_crops": ["Vegetables", "Flowers", "Young Plants"],
                "valid_until": datetime.utcnow() + timedelta(hours=12),
                "acknowledged_by": [],
                "timestamp": datetime.utcnow()
            }
            result = await db.weather_alerts.insert_one(alert_doc)
            alerts.append(str(result.inserted_id))
    
    # Heavy rainfall alert
    if preferences.get("enable_rainfall_alerts", True) and "Rain" in conditions:
        alert_key = f"rainfall_{location}_{datetime.utcnow().date()}"
        
        existing = await db.weather_alerts.find_one({
            "alert_key": alert_key,
            "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        if not existing:
            severity = "high" if rainfall > 20 else "medium"
            alert_doc = {
                "alert_key": alert_key,
                "location": location,
                "alert_type": "rainfall",
                "message": f"Rainfall expected. Current conditions: {conditions}",
                "recommended_action": "Postpone pesticide/fertilizer application, ensure proper drainage, and monitor for waterlogging.",
                "severity": severity,
                "affected_crops": ["All Crops"],
                "valid_until": datetime.utcnow() + timedelta(hours=6),
                "acknowledged_by": [],
                "timestamp": datetime.utcnow()
            }
            result = await db.weather_alerts.insert_one(alert_doc)
            alerts.append(str(result.inserted_id))
    
    # High humidity alert
    if humidity > 85:
        alert_key = f"humidity_{location}_{datetime.utcnow().date()}"
        
        existing = await db.weather_alerts.find_one({
            "alert_key": alert_key,
            "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        if not existing:
            alert_doc = {
                "alert_key": alert_key,
                "location": location,
                "alert_type": "humidity",
                "message": f"Very high humidity: {humidity}%. Risk of fungal diseases.",
                "recommended_action": "Monitor crops for fungal diseases, ensure good air circulation, and consider preventive organic sprays.",
                "severity": "medium",
                "affected_crops": ["All Crops"],
                "valid_until": datetime.utcnow() + timedelta(hours=12),
                "acknowledged_by": [],
                "timestamp": datetime.utcnow()
            }
            result = await db.weather_alerts.insert_one(alert_doc)
            alerts.append(str(result.inserted_id))
    
    # Strong wind alert
    if preferences.get("enable_storm_alerts", True) and wind_speed > 30:
        alert_key = f"wind_{location}_{datetime.utcnow().date()}"
        
        existing = await db.weather_alerts.find_one({
            "alert_key": alert_key,
            "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        if not existing:
            severity = "high" if wind_speed > 50 else "medium"
            alert_doc = {
                "alert_key": alert_key,
                "location": location,
                "alert_type": "wind",
                "message": f"Strong winds: {round(wind_speed)} km/h. Secure crops and structures.",
                "recommended_action": "Provide support to tall crops, avoid spraying operations, and check greenhouse structures.",
                "severity": severity,
                "affected_crops": ["Tall Crops", "Young Plants"],
                "valid_until": datetime.utcnow() + timedelta(hours=6),
                "acknowledged_by": [],
                "timestamp": datetime.utcnow()
            }
            result = await db.weather_alerts.insert_one(alert_doc)
            alerts.append(str(result.inserted_id))
    
    return alerts


# ============= Weather API Endpoints =============

@app.get("/api/weather/current")
async def get_current_weather(current_user: dict = Depends(get_current_user)):
    """Get current weather conditions for user's location"""
    
    location = current_user.get("district", "Visakhapatnam")
    
    weather_data = await fetch_openweather_current(location)
    
    if not weather_data:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    
    # Get user preferences for alert thresholds
    preferences = await db.weather_preferences.find_one({"user_id": str(current_user["_id"])})
    if not preferences:
        preferences = {}
    
    # Generate automatic alerts based on conditions
    await generate_weather_alerts(location, weather_data, preferences)
    
    current_weather = {
        "location": location,
        "temperature": round(weather_data["main"]["temp"]),
        "feels_like": round(weather_data["main"]["feels_like"]),
        "temp_min": round(weather_data["main"]["temp_min"]),
        "temp_max": round(weather_data["main"]["temp_max"]),
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": round(weather_data["wind"]["speed"] * 3.6),
        "wind_direction": weather_data["wind"].get("deg", 0),
        "cloudiness": weather_data["clouds"]["all"],
        "conditions": weather_data["weather"][0]["main"],
        "description": weather_data["weather"][0]["description"],
        "icon": weather_data["weather"][0]["icon"],
        "visibility": weather_data.get("visibility", 0) / 1000,
        "rainfall": weather_data.get("rain", {}).get("1h", 0),
        "sunrise": datetime.fromtimestamp(weather_data["sys"]["sunrise"]).isoformat(),
        "sunset": datetime.fromtimestamp(weather_data["sys"]["sunset"]).isoformat(),
        "last_updated": datetime.utcnow().isoformat()
    }
    
    return current_weather


@app.get("/api/weather/forecast")
async def get_weather_forecast(
    days: int = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get weather forecast for next N days"""
    
    location = current_user.get("district", "Visakhapatnam")
    
    forecast_data = await fetch_openweather_forecast(location)
    
    if not forecast_data:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    
    daily_forecast = {}
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for item in forecast_data["list"]:
        date = datetime.fromtimestamp(item["dt"])
        day_key = date.strftime("%Y-%m-%d")
        
        if day_key not in daily_forecast:
            daily_forecast[day_key] = {
                "date": day_key,
                "day": days_of_week[date.weekday()],
                "temps": [],
                "conditions": [],
                "rainfall": 0,
                "humidity": [],
                "wind_speed": []
            }
        
        daily_forecast[day_key]["temps"].append(item["main"]["temp"])
        daily_forecast[day_key]["conditions"].append(item["weather"][0]["main"])
        daily_forecast[day_key]["rainfall"] += item.get("rain", {}).get("3h", 0)
        daily_forecast[day_key]["humidity"].append(item["main"]["humidity"])
        daily_forecast[day_key]["wind_speed"].append(item["wind"]["speed"])
    
    forecast = []
    for day_key in sorted(daily_forecast.keys())[:days]:
        day_data = daily_forecast[day_key]
        forecast.append({
            "date": day_data["date"],
            "day": day_data["day"][:3],
            "temp_min": round(min(day_data["temps"])),
            "temp_max": round(max(day_data["temps"])),
            "temp_avg": round(sum(day_data["temps"]) / len(day_data["temps"])),
            "conditions": max(set(day_data["conditions"]), key=day_data["conditions"].count),
            "rainfall_mm": round(day_data["rainfall"], 1),
            "humidity": round(sum(day_data["humidity"]) / len(day_data["humidity"])),
            "wind_speed": round(sum(day_data["wind_speed"]) / len(day_data["wind_speed"]) * 3.6)
        })
    
    return {
        "location": location,
        "forecast": forecast
    }


@app.get("/api/weather/alerts/active")
async def get_active_weather_alerts(current_user: dict = Depends(get_current_user)):
    """Get currently active weather alerts for user's location"""
    
    location = current_user.get("district", "Visakhapatnam")
    now = datetime.utcnow()
    user_id = str(current_user["_id"])
    
    print(f"Fetching alerts for location: {location}, user: {user_id}")
    
    # Find active alerts that haven't been acknowledged by this user
    alerts = await db.weather_alerts.find({
        "location": {"$regex": location, "$options": "i"},
        "$or": [
            {"valid_until": {"$gte": now}},
            {"valid_until": None}
        ],
        "acknowledged_by": {"$ne": user_id}  # Not acknowledged by this user
    }).sort("timestamp", -1).to_list(20)
    
    print(f"Found {len(alerts)} alerts")
    
    result = []
    for a in alerts:
        alert_id = str(a["_id"])
        alert_data = {
            "id": alert_id,
            "type": a.get("alert_type", "general"),
            "severity": a.get("severity", "medium"),
            "message": a.get("message", ""),
            "recommended_action": a.get("recommended_action", ""),
            "affected_crops": a.get("affected_crops", []),
            "timestamp": a["timestamp"].isoformat()
        }
        result.append(alert_data)
    
    print(f"Returning {len(result)} alerts with IDs: {[a['id'] for a in result]}")
    return result


@app.get("/api/weather/advisory")
async def get_weather_based_advisory(current_user: dict = Depends(get_current_user)):
    """Get farming advisory based on current weather"""
    
    location = current_user.get("district", "Visakhapatnam")
    
    weather_data = await fetch_openweather_current(location)
    
    if not weather_data:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    conditions = weather_data["weather"][0]["main"]
    rainfall = weather_data.get("rain", {}).get("1h", 0)
    wind_speed = weather_data["wind"]["speed"] * 3.6
    
    advisories = []
    
    # Temperature advisory
    if temp > 35:
        advisories.append({
            "type": "temperature",
            "priority": "high",
            "message": f"High temperature alert: {round(temp)}°C",
            "recommendations": [
                "Increase irrigation frequency, especially for young plants",
                "Apply organic mulch to conserve soil moisture",
                "Provide shade nets for sensitive vegetables",
                "Water crops early morning or late evening"
            ]
        })
    elif temp < 15:
        advisories.append({
            "type": "temperature",
            "priority": "medium",
            "message": f"Low temperature alert: {round(temp)}°C",
            "recommendations": [
                "Protect frost-sensitive crops",
                "Consider using plastic covers for seedlings",
                "Delay irrigation if frost is expected"
            ]
        })
    
    # Humidity advisory
    if humidity > 80:
        advisories.append({
            "type": "humidity",
            "priority": "medium",
            "message": f"High humidity: {humidity}%",
            "recommendations": [
                "Monitor for fungal diseases closely",
                "Ensure good air circulation in crops",
                "Avoid excessive irrigation",
                "Consider preventive neem spray application"
            ]
        })
    
    # Rainfall advisory
    if "Rain" in conditions or rainfall > 0:
        advisories.append({
            "type": "rainfall",
            "priority": "high",
            "message": "Rain expected or occurring",
            "recommendations": [
                "Postpone pesticide and fertilizer application",
                "Ensure proper drainage to prevent waterlogging",
                "Good time for transplanting rice seedlings",
                "Monitor for pest and disease outbreaks after rain"
            ]
        })
    
    # Wind advisory
    if wind_speed > 30:
        advisories.append({
            "type": "wind",
            "priority": "medium",
            "message": f"Strong winds: {round(wind_speed)} km/h",
            "recommendations": [
                "Provide support to tall crops",
                "Avoid spraying operations",
                "Check greenhouse structures"
            ]
        })
    
    return {
        "current_weather": {
            "temperature": round(temp),
            "humidity": humidity,
            "conditions": conditions,
            "rainfall": rainfall,
            "wind_speed": round(wind_speed)
        },
        "advisories": advisories
    }


@app.get("/api/weather/preferences")
async def get_weather_preferences(current_user: dict = Depends(get_current_user)):
    """Get user's weather notification preferences"""
    
    preferences = await db.weather_preferences.find_one({"user_id": str(current_user["_id"])})
    
    if not preferences:
        return {
            "enable_weather_alerts": True,
            "enable_rainfall_alerts": True,
            "enable_temperature_alerts": True,
            "enable_storm_alerts": True,
            "high_temp_threshold": 35.0,
            "low_temp_threshold": 15.0,
            "heavy_rain_threshold": 50.0,
            "morning_alert": True,
            "evening_alert": True,
            "primary_location": current_user.get("district", "Visakhapatnam"),
            "additional_locations": []
        }
    
    return {
        "enable_weather_alerts": preferences.get("enable_weather_alerts", True),
        "enable_rainfall_alerts": preferences.get("enable_rainfall_alerts", True),
        "enable_temperature_alerts": preferences.get("enable_temperature_alerts", True),
        "enable_storm_alerts": preferences.get("enable_storm_alerts", True),
        "high_temp_threshold": preferences.get("high_temp_threshold", 35.0),
        "low_temp_threshold": preferences.get("low_temp_threshold", 15.0),
        "heavy_rain_threshold": preferences.get("heavy_rain_threshold", 50.0),
        "morning_alert": preferences.get("morning_alert", True),
        "evening_alert": preferences.get("evening_alert", True),
        "primary_location": preferences.get("primary_location", current_user.get("district", "Visakhapatnam")),
        "additional_locations": preferences.get("additional_locations", [])
    }


@app.put("/api/weather/preferences")
async def update_weather_preferences(
    preferences_data: WeatherPreferencesUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update user's weather notification preferences"""
    
    user_id = str(current_user["_id"])
    
    existing = await db.weather_preferences.find_one({"user_id": user_id})
    
    update_data = preferences_data.dict(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()
    
    if existing:
        await db.weather_preferences.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
    else:
        update_data["user_id"] = user_id
        update_data["created_at"] = datetime.utcnow()
        await db.weather_preferences.insert_one(update_data)
    
    return {"message": "Weather preferences updated successfully"}


@app.post("/api/weather/alerts/{alert_id}/acknowledge")
async def acknowledge_weather_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark weather alert as acknowledged by user"""
    
    user_id = str(current_user["_id"])
    
    print(f"Acknowledging alert {alert_id} for user {user_id}")
    
    try:
        alert = await db.weather_alerts.find_one({"_id": ObjectId(alert_id)})
    except Exception as e:
        print(f"Invalid alert ID: {e}")
        raise HTTPException(status_code=400, detail="Invalid alert ID format")
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    await db.weather_alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {"$addToSet": {"acknowledged_by": user_id}}
    )
    
    print(f"Alert {alert_id} acknowledged successfully")
    return {"message": "Alert acknowledged successfully"}


@app.get("/api/weather/statistics")
async def get_weather_statistics(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get weather statistics for user's location"""
    
    location = current_user.get("district", "Visakhapatnam")
    start_date = datetime.utcnow() - timedelta(days=days)
    
    alerts_pipeline = [
        {
            "$match": {
                "location": {"$regex": location, "$options": "i"},
                "timestamp": {"$gte": start_date}
            }
        },
        {
            "$group": {
                "_id": "$alert_type",
                "count": {"$sum": 1}
            }
        }
    ]
    
    alerts_by_type = await db.weather_alerts.aggregate(alerts_pipeline).to_list(100)
    
    return {
        "period": f"Last {days} days",
        "location": location,
        "alerts_by_type": {item["_id"]: item["count"] for item in alerts_by_type},
        "total_alerts": sum(item["count"] for item in alerts_by_type)
    }
#  consulation 
# Enhanced WebSocket Manager for Consultations

# ============= WebSocket Connection Manager =============
class ConsultationManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_sessions: Dict[str, Set[str]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        for session_users in self.active_sessions.values():
            session_users.discard(user_id)

    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except:
                self.disconnect(user_id)

    async def broadcast_to_session(self, session_id: str, message: dict, exclude_user: str = None):
        if session_id in self.active_sessions:
            for user_id in self.active_sessions[session_id]:
                if user_id != exclude_user:
                    await self.send_to_user(user_id, message)

consultation_manager = ConsultationManager()


# ============= SPECIALIST PROFILE MANAGEMENT =============

@app.post("/api/specialists/profile")
async def create_or_update_specialist_profile(
    profile_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create or update specialist profile"""
    
    if current_user.get("role") != "specialist":
        raise HTTPException(
            status_code=403,
            detail="Only users with specialist role can create specialist profiles"
        )
    
    user_id = str(current_user["_id"])
    existing = await db.specialist_profiles.find_one({"user_id": user_id})
    
    profile_doc = {
        "user_id": user_id,
        "specialization": profile_data.get("specialization", ["General Agriculture"]),
        "experience_years": profile_data.get("experience_years", 0),
        "qualification": profile_data.get("qualification", "Agricultural Expert"),
        "languages": profile_data.get("languages", ["telugu", "english"]),
        "crops_expertise": profile_data.get("crops_expertise", []),
        "diseases_expertise": profile_data.get("diseases_expertise", []),
        "bio": profile_data.get("bio", "Agricultural specialist ready to help farmers"),
        "consultation_fee": profile_data.get("consultation_fee", 0.0),
        "is_online": profile_data.get("is_online", False),
        "last_active": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    if existing:
        await db.specialist_profiles.update_one(
            {"user_id": user_id},
            {"$set": profile_doc}
        )
        return {
            "message": "Profile updated successfully",
            "profile_id": str(existing["_id"])
        }
    else:
        profile_doc["created_at"] = datetime.utcnow()
        profile_doc["average_rating"] = 0.0
        profile_doc["total_consultations"] = 0
        profile_doc["total_ratings"] = 0
        result = await db.specialist_profiles.insert_one(profile_doc)
        return {
            "message": "Profile created successfully",
            "profile_id": str(result.inserted_id)
        }


@app.get("/api/specialists/available")
async def get_available_specialists(
    specialization: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get list of specialists - ONLY ACCESSIBLE BY FARMERS"""
    
    if current_user.get("role") != "farmer":
        raise HTTPException(
            status_code=403,
            detail="Only farmers can view specialists list"
        )
    
    query = {}
    if specialization:
        query["specialization"] = {"$in": [specialization]}
    
    specialists = await db.specialist_profiles.find(query).to_list(100)
    
    result = []
    for spec in specialists:
        user = await db.users.find_one({"_id": ObjectId(spec["user_id"])})
        if user and user.get("role") == "specialist":
            result.append({
                "id": spec["user_id"],
                "name": user["name"],
                "email": user.get("email"),
                "specialization": spec.get("specialization", []),
                "experience_years": spec.get("experience_years", 0),
                "languages": spec.get("languages", []),
                "average_rating": spec.get("average_rating", 0.0),
                "total_consultations": spec.get("total_consultations", 0),
                "bio": spec.get("bio"),
                "is_online": spec.get("is_online", False),
                "last_active": spec.get("last_active").isoformat() if spec.get("last_active") else None
            })
    
    result.sort(key=lambda x: (not x['is_online'], -x['average_rating']))
    return result


@app.put("/api/specialists/status")
async def update_online_status(
    status_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update specialist online status"""
    
    user_id = str(current_user["_id"])
    is_online = status_data.get("is_online", False)
    
    profile = await db.specialist_profiles.find_one({"user_id": user_id})
    
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Specialist profile not found. Please create a profile first."
        )
    
    await db.specialist_profiles.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "is_online": is_online,
                "last_active": datetime.utcnow()
            }
        }
    )
    
    if is_online:
        pending_requests = await db.consultation_sessions.find({
            "specialist_id": user_id,
            "session_type": "video",
            "status": "pending",
            "expires_at": {"$gte": datetime.utcnow()}
        }).to_list(100)
        
        pending_count = len(pending_requests)
        
        if pending_count > 0:
            await consultation_manager.send_to_user(
                user_id,
                {
                    "type": "pending_requests_notification",
                    "count": pending_count,
                    "message": f"You have {pending_count} pending video call request(s)"
                }
            )
        
        return {
            "message": "Status updated successfully",
            "is_online": is_online,
            "pending_requests": pending_count
        }
    
    return {
        "message": "Status updated successfully",
        "is_online": is_online
    }


@app.get("/api/specialists/my-profile")
async def get_my_specialist_profile(
    current_user: dict = Depends(get_current_user)
):
    """Get current user's specialist profile"""
    
    profile = await db.specialist_profiles.find_one({"user_id": str(current_user["_id"])})
    
    if not profile:
        return {
            "exists": False,
            "message": "No specialist profile found"
        }
    
    return {
        "exists": True,
        "profile": {
            "user_id": profile["user_id"],
            "specialization": profile.get("specialization", []),
            "experience_years": profile.get("experience_years", 0),
            "qualification": profile.get("qualification"),
            "languages": profile.get("languages", []),
            "bio": profile.get("bio"),
            "is_online": profile.get("is_online", False),
            "average_rating": profile.get("average_rating", 0.0),
            "total_consultations": profile.get("total_consultations", 0)
        }
    }


# ============= DIRECT CHAT (NO REQUEST NEEDED) =============

@app.post("/api/consultations/start-chat/{specialist_id}")
async def start_direct_chat(
    specialist_id: str,
    chat_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Start direct chat with any specialist - NO REQUEST OR ACCEPTANCE NEEDED"""
    
    specialist = await db.users.find_one({"_id": ObjectId(specialist_id)})
    if not specialist or specialist.get("role") != "specialist":
        raise HTTPException(status_code=404, detail="Specialist not found")
    
    # Check for existing active chat
    existing_chat = await db.consultation_sessions.find_one({
        "farmer_id": str(current_user["_id"]),
        "specialist_id": specialist_id,
        "session_type": "chat",
        "status": "active"
    })
    
    if existing_chat:
        return {
            "session_id": str(existing_chat["_id"]),
            "room_id": existing_chat.get("room_id"),
            "message": "Existing chat session found",
            "existing": True
        }
    
    room_id = str(uuid.uuid4())
    
    session_doc = {
        "farmer_id": str(current_user["_id"]),
        "farmer_name": current_user["name"],
        "specialist_id": specialist_id,
        "specialist_name": specialist["name"],
        "session_type": "chat",
        "status": "active",
        "topic": chat_data.get("topic", f"Chat with {specialist['name']}"),
        "description": chat_data.get("description"),
        "room_id": room_id,
        "messages": [],
        "shared_photos": [],
        "started_at": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.consultation_sessions.insert_one(session_doc)
    
    # Notify specialist
    await consultation_manager.send_to_user(
        specialist_id,
        {
            "type": "new_chat_session",
            "session_id": str(result.inserted_id),
            "farmer_name": current_user["name"],
            "topic": chat_data.get("topic", "General Consultation"),
            "room_id": room_id
        }
    )
    
    return {
        "session_id": str(result.inserted_id),
        "room_id": room_id,
        "message": "Chat session started",
        "existing": False
    }


# ============= VIDEO CALL REQUESTS (REQUIRES ACCEPTANCE) =============

@app.post("/api/consultations/request-video/{specialist_id}")
async def request_video_call(
    specialist_id: str,
    request_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Request video call - REQUIRES specialist acceptance"""
    
    specialist = await db.users.find_one({"_id": ObjectId(specialist_id)})
    if not specialist or specialist.get("role") != "specialist":
        raise HTTPException(status_code=404, detail="Specialist not found")
    
    specialist_profile = await db.specialist_profiles.find_one({"user_id": specialist_id})
    
    room_id = str(uuid.uuid4())
    
    session_doc = {
        "farmer_id": str(current_user["_id"]),
        "farmer_name": current_user["name"],
        "farmer_phone": current_user.get("phone"),
        "specialist_id": specialist_id,
        "specialist_name": specialist["name"],
        "session_type": "video",
        "status": "pending",
        "topic": request_data.get("topic", "Video Consultation"),
        "description": request_data.get("description"),
        "urgency": request_data.get("urgency", "normal"),
        "related_crop_photo_id": request_data.get("related_crop_photo_id"),
        "room_id": room_id,
        "messages": [],
        "shared_photos": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=24)
    }
    
    result = await db.consultation_sessions.insert_one(session_doc)
    
    # Notify specialist if online
    if specialist_profile and specialist_profile.get("is_online"):
        await consultation_manager.send_to_user(
            specialist_id,
            {
                "type": "video_call_request",
                "request_id": str(result.inserted_id),
                "farmer_name": current_user["name"],
                "topic": request_data.get("topic"),
                "urgency": request_data.get("urgency", "normal")
            }
        )
    
    return {
        "request_id": str(result.inserted_id),
        "message": f"Video call request sent to {specialist['name']}",
        "specialist_online": specialist_profile.get("is_online", False) if specialist_profile else False,
        "expires_at": session_doc["expires_at"].isoformat()
    }


@app.get("/api/consultations/video-requests/pending")
async def get_pending_video_requests(
    current_user: dict = Depends(require_role("specialist"))
):
    """Get pending video requests for specialist"""
    
    user_id = str(current_user["_id"])
    now = datetime.utcnow()
    
    requests = await db.consultation_sessions.find({
        "specialist_id": user_id,
        "session_type": "video",
        "status": "pending",
        "expires_at": {"$gte": now}
    }).sort("created_at", -1).to_list(100)
    
    return [
        {
            "id": str(r["_id"]),
            "farmer_name": r["farmer_name"],
            "farmer_phone": r.get("farmer_phone"),
            "topic": r["topic"],
            "description": r.get("description"),
            "urgency": r.get("urgency", "normal"),
            "room_id": r.get("room_id"),
            "created_at": r["created_at"].isoformat(),
            "expires_at": r["expires_at"].isoformat()
        } for r in requests
    ]


@app.post("/api/consultations/{session_id}/accept-video")
async def accept_video_request(
    session_id: str,
    current_user: dict = Depends(require_role("specialist"))
):
    """Specialist accepts video call request"""
    
    session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Request not found")
    
    if session["specialist_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not your request")
    
    if session["status"] != "pending":
        raise HTTPException(status_code=400, detail="Request is not pending")
    
    if session.get("session_type") != "video":
        raise HTTPException(status_code=400, detail="Not a video call request")
    
    await db.consultation_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "status": "accepted",
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    # Notify farmer
    await consultation_manager.send_to_user(
        session["farmer_id"],
        {
            "type": "video_request_accepted",
            "session_id": session_id,
            "specialist_name": current_user["name"],
            "message": f"{current_user['name']} accepted your video call request!",
            "room_id": session.get("room_id")
        }
    )
    
    return {
        "message": "Video call request accepted",
        "session_id": session_id,
        "room_id": session.get("room_id")
    }


@app.get("/api/consultations/video-requests/my-requests")
async def get_my_video_requests(
    current_user: dict = Depends(get_current_user)
):
    """Get farmer's video call requests"""
    
    if current_user.get("role") != "farmer":
        raise HTTPException(status_code=403, detail="Only farmers can access this")
    
    user_id = str(current_user["_id"])
    
    requests = await db.consultation_sessions.find({
        "farmer_id": user_id,
        "session_type": "video",
        "status": {"$in": ["pending", "accepted"]}
    }).sort("created_at", -1).to_list(100)
    
    return [
        {
            "id": str(r["_id"]),
            "specialist_name": r["specialist_name"],
            "topic": r["topic"],
            "description": r.get("description"),
            "status": r["status"],
            "room_id": r.get("room_id"),
            "created_at": r["created_at"].isoformat(),
            "expires_at": r["expires_at"].isoformat() if r.get("expires_at") else None
        } for r in requests
    ]


# ============= GET ACTIVE CHATS =============

@app.get("/api/consultations/chats/active")
async def get_active_chats(
    current_user: dict = Depends(get_current_user)
):
    """Get active chat sessions"""
    
    user_id = str(current_user["_id"])
    
    query = {
        "session_type": "chat",
        "status": "active"
    }
    
    if current_user.get("role") == "specialist":
        query["specialist_id"] = user_id
    else:
        query["farmer_id"] = user_id
    
    chats = await db.consultation_sessions.find(query).sort(
        "updated_at", -1
    ).to_list(100)
    
    result = []
    for chat in chats:
        latest_msg = await db.consultation_messages.find_one(
            {"session_id": str(chat["_id"])},
            sort=[("timestamp", -1)]
        )
        
        unread_count = await db.consultation_messages.count_documents({
            "session_id": str(chat["_id"]),
            "sender_id": {"$ne": user_id},
            "read": False
        })
        
        result.append({
            "id": str(chat["_id"]),
            "room_id": chat.get("room_id"),
            "farmer_name": chat["farmer_name"],
            "specialist_name": chat["specialist_name"],
            "topic": chat["topic"],
            "started_at": chat["started_at"].isoformat() if chat.get("started_at") else None,
            "last_message_text": latest_msg.get("message_text") if latest_msg else None,
            "last_message_time": latest_msg["timestamp"].isoformat() if latest_msg else None,
            "unread_count": unread_count
        })
    
    return result


# ============= START VIDEO CALL (ONLY IF ACCEPTED) =============

@app.post("/api/consultations/{session_id}/start-video")
async def start_video_call(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Start video call - only works if request was accepted"""
    
    session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_id = str(current_user["_id"])
    if user_id not in [session["farmer_id"], session.get("specialist_id")]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if session.get("session_type") != "video":
        raise HTTPException(status_code=400, detail="Not a video session")
    
    if session["status"] != "accepted":
        raise HTTPException(
            status_code=400, 
            detail=f"Video call not accepted yet. Status: {session['status']}"
        )
    
    await db.consultation_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "status": "active",
                "started_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    if session_id not in consultation_manager.active_sessions:
        consultation_manager.active_sessions[session_id] = set()
    consultation_manager.active_sessions[session_id].add(user_id)
    
    return {
        "message": "Video call started",
        "room_id": session["room_id"],
        "session_type": "video"
    }


# ============= END SESSION =============

@app.post("/api/consultations/{session_id}/end")
async def end_consultation(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """End consultation session"""
    
    session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_id = str(current_user["_id"])
    if user_id not in [session["farmer_id"], session.get("specialist_id")]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    duration = None
    if session.get("started_at"):
        duration = int((datetime.utcnow() - session["started_at"]).total_seconds() / 60)
    
    await db.consultation_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "status": "completed",
                "ended_at": datetime.utcnow(),
                "duration_minutes": duration,
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    if session_id in consultation_manager.active_sessions:
        del consultation_manager.active_sessions[session_id]
    
    if session.get("specialist_id"):
        await db.specialist_profiles.update_one(
            {"user_id": session["specialist_id"]},
            {"$inc": {"total_consultations": 1}}
        )
    
    return {
        "message": "Session ended successfully",
        "duration_minutes": duration
    }


# ============= MESSAGES =============

@app.post("/api/consultations/{session_id}/messages")
async def send_consultation_message(
    session_id: str,
    message_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Send message in consultation session"""
    
    session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_id = str(current_user["_id"])
    if user_id not in [session["farmer_id"], session.get("specialist_id")]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    sender_role = "farmer" if user_id == session["farmer_id"] else "specialist"
    
    message_doc = {
        "session_id": session_id,
        "sender_id": user_id,
        "sender_name": current_user["name"],
        "sender_role": sender_role,
        "message_type": message_data.get("message_type", "text"),
        "message_text": message_data.get("message_text"),
        "media_url": message_data.get("media_url"),
        "read": False,
        "timestamp": datetime.utcnow()
    }
    
    result = await db.consultation_messages.insert_one(message_doc)
    
    # Update session last activity
    await db.consultation_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updated_at": datetime.utcnow()}}
    )
    
    # Broadcast to other participant
    await consultation_manager.broadcast_to_session(
        session_id,
        {
            "type": "new_message",
            "message": {
                "id": str(result.inserted_id),
                "sender_name": current_user["name"],
                "sender_role": sender_role,
                "message_type": message_data.get("message_type", "text"),
                "message_text": message_data.get("message_text"),
                "timestamp": datetime.utcnow().isoformat()
            }
        },
        exclude_user=user_id
    )
    
    return {
        "message_id": str(result.inserted_id),
        "message": "Message sent successfully"
    }


@app.get("/api/consultations/{session_id}/messages")
async def get_consultation_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all messages from consultation session"""
    
    session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_id = str(current_user["_id"])
    if user_id not in [session["farmer_id"], session.get("specialist_id")]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    messages = await db.consultation_messages.find(
        {"session_id": session_id}
    ).sort("timestamp", 1).to_list(1000)
    
    return [
        {
            "id": str(m["_id"]),
            "sender_name": m["sender_name"],
            "sender_role": m["sender_role"],
            "message_type": m["message_type"],
            "message_text": m.get("message_text"),
            "media_url": m.get("media_url"),
            "timestamp": m["timestamp"].isoformat()
        } for m in messages
    ]


# ============= WebSocket =============

@app.websocket("/ws/consultation/{session_id}")
async def consultation_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str
):
    """WebSocket endpoint for real-time consultation"""
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        
        if not user_id:
            await websocket.close(code=1008)
            return
        
        session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
        if not session or user_id not in [session["farmer_id"], session.get("specialist_id")]:
            await websocket.close(code=1008)
            return
        
        await consultation_manager.connect(user_id, websocket)
        
        if session_id not in consultation_manager.active_sessions:
            consultation_manager.active_sessions[session_id] = set()
        consultation_manager.active_sessions[session_id].add(user_id)
        
        try:
            while True:
                data = await websocket.receive_json()
                
                if data["type"] == "webrtc_signal":
                    await consultation_manager.broadcast_to_session(
                        session_id,
                        data,
                        exclude_user=user_id
                    )
                elif data["type"] == "typing":
                    await consultation_manager.broadcast_to_session(
                        session_id,
                        {"type": "typing", "user_id": user_id},
                        exclude_user=user_id
                    )
        
        except WebSocketDisconnect:
            consultation_manager.disconnect(user_id)
            if session_id in consultation_manager.active_sessions:
                consultation_manager.active_sessions[session_id].discard(user_id)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011)


# ============= WebRTC Signaling =============

@app.post("/api/consultations/{session_id}/webrtc-signal")
async def send_webrtc_signal(
    session_id: str,
    signal_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Send WebRTC signaling data"""
    
    session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_id = str(current_user["_id"])
    signal_type = signal_data.get("signal_type")
    signal_payload = signal_data.get("signal_data")
    
    if signal_type == "offer":
        await db.consultation_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"webrtc_offer": signal_payload}}
        )
    elif signal_type == "answer":
        await db.consultation_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"webrtc_answer": signal_payload}}
        )
    elif signal_type == "ice_candidate":
        await db.consultation_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$push": {"ice_candidates": signal_payload}}
        )
    
    await consultation_manager.broadcast_to_session(
        session_id,
        {
            "type": "webrtc_signal",
            "signal_type": signal_type,
            "signal_data": signal_payload,
            "from_user": user_id
        },
        exclude_user=user_id
    )
    
    return {"message": "Signal sent successfully"}


# ============= STATISTICS =============

@app.get("/api/consultations/statistics")
async def get_consultation_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get consultation statistics for current user"""
    
    user_id = str(current_user["_id"])
    
    if current_user.get("role") == "specialist":
        total_consultations = await db.consultation_sessions.count_documents({
            "specialist_id": user_id,
            "status": "completed"
        })
        
        active_consultations = await db.consultation_sessions.count_documents({
            "specialist_id": user_id,
            "status": {"$in": ["accepted", "active"]}
        })
        
        profile = await db.specialist_profiles.find_one({"user_id": user_id})
        
        return {
            "role": "specialist",
            "total_consultations": total_consultations,
            "active_consultations": active_consultations,
            "average_rating": profile.get("average_rating", 0) if profile else 0,
            "is_online": profile.get("is_online", False) if profile else False
        }
    else:
        total_consultations = await db.consultation_sessions.count_documents({
            "farmer_id": user_id
        })
        
        active_consultations = await db.consultation_sessions.count_documents({
            "farmer_id": user_id,
            "status": {"$in": ["pending", "accepted", "active"]}
        })
        
        completed_consultations = await db.consultation_sessions.count_documents({
            "farmer_id": user_id,
            "status": "completed"
        })
        
        return {
            "role": "farmer",
            "total_consultations": total_consultations,
            "active_consultations": active_consultations,
            "completed_consultations": completed_consultations
        }

# Add to your main.py

@app.get("/api/consultations/active-calls")
async def get_active_calls(
    current_user: dict = Depends(require_role("specialist"))
):
    """Get active video calls waiting for specialist to join"""
    
    user_id = str(current_user["_id"])
    
    # Find calls that are active but specialist hasn't joined yet
    active_calls = await db.consultation_sessions.find({
        "specialist_id": user_id,
        "session_type": "video",
        "status": "active",
        "started_at": {"$exists": True}
    }).sort("started_at", -1).to_list(10)
    
    return [
        {
            "id": str(call["_id"]),
            "farmer_name": call["farmer_name"],
            "topic": call["topic"],
            "room_id": call.get("room_id"),
            "started_at": call["started_at"].isoformat()
        } for call in active_calls
    ]


@app.post("/api/consultations/{session_id}/join-video")
async def join_video_call(
    session_id: str,
    current_user: dict = Depends(require_role("specialist"))
):
    """Specialist joins an active video call"""
    
    session = await db.consultation_sessions.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["specialist_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not your session")
    
    if session.get("session_type") != "video":
        raise HTTPException(status_code=400, detail="Not a video session")
    
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Call is not active")
    
    return {
        "message": "Joining video call",
        "room_id": session["room_id"],
        "session_type": "video"
    }
#  suppliers 
# Add these endpoints to your main.py file

# ============= SUPPLIERS MANAGEMENT =============

@app.post("/api/suppliers")
async def create_supplier(
    supplier_data: dict,
    current_user: dict = Depends(require_role("specialist"))
):
    """Create a new supplier (specialists only)"""
    
    supplier_doc = {
        "name": supplier_data["name"],
        "location": supplier_data["location"],
        "contact_info": {
            "phone": supplier_data.get("phone"),
            "email": supplier_data.get("email"),
            "address": supplier_data.get("address"),
            "website": supplier_data.get("website")
        },
        "products_available": supplier_data.get("products_available", []),
        "crop_association": supplier_data.get("crop_association", []),
        "rating": 0.0,
        "total_ratings": 0,
        "verified": False,
        "created_by": str(current_user["_id"]),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.suppliers.insert_one(supplier_doc)
    
    return {
        "supplier_id": str(result.inserted_id),
        "message": "Supplier added successfully"
    }


@app.get("/api/suppliers")
async def get_suppliers(
    location: Optional[str] = None,
    product: Optional[str] = None,
    crop: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get list of suppliers with filters"""
    
    query = {}
    
    if location:
        query["location"] = {"$regex": location, "$options": "i"}
    
    if product:
        query["products_available"] = {"$in": [product]}
    
    if crop:
        query["crop_association"] = {"$in": [crop]}
    
    suppliers = await db.suppliers.find(query).sort("name", 1).to_list(1000)
    
    return [
        {
            "id": str(s["_id"]),
            "name": s["name"],
            "location": s["location"],
            "contact_info": s["contact_info"],
            "products_available": s["products_available"],
            "crop_association": s.get("crop_association", []),
            "rating": s.get("rating", 0.0),
            "verified": s.get("verified", False),
            "created_at": s["created_at"].isoformat()
        } for s in suppliers
    ]


@app.get("/api/suppliers/{supplier_id}")
async def get_supplier_detail(
    supplier_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed supplier information"""
    
    supplier = await db.suppliers.find_one({"_id": ObjectId(supplier_id)})
    
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    
    # Get ratings/reviews if implemented
    reviews = await db.supplier_reviews.find(
        {"supplier_id": supplier_id}
    ).sort("created_at", -1).limit(10).to_list(10)
    
    return {
        "id": str(supplier["_id"]),
        "name": supplier["name"],
        "location": supplier["location"],
        "contact_info": supplier["contact_info"],
        "products_available": supplier["products_available"],
        "crop_association": supplier.get("crop_association", []),
        "rating": supplier.get("rating", 0.0),
        "total_ratings": supplier.get("total_ratings", 0),
        "verified": supplier.get("verified", False),
        "reviews": [
            {
                "user_name": r.get("user_name"),
                "rating": r["rating"],
                "review": r.get("review"),
                "created_at": r["created_at"].isoformat()
            } for r in reviews
        ],
        "created_at": supplier["created_at"].isoformat()
    }


@app.put("/api/suppliers/{supplier_id}")
async def update_supplier(
    supplier_id: str,
    update_data: dict,
    current_user: dict = Depends(require_role("specialist"))
):
    """Update supplier information"""
    
    supplier = await db.suppliers.find_one({"_id": ObjectId(supplier_id)})
    
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    
    if supplier["created_by"] != str(current_user["_id"]) and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_fields = {}
    
    if "name" in update_data:
        update_fields["name"] = update_data["name"]
    if "location" in update_data:
        update_fields["location"] = update_data["location"]
    if "contact_info" in update_data:
        update_fields["contact_info"] = update_data["contact_info"]
    if "products_available" in update_data:
        update_fields["products_available"] = update_data["products_available"]
    if "crop_association" in update_data:
        update_fields["crop_association"] = update_data["crop_association"]
    
    update_fields["updated_at"] = datetime.utcnow()
    
    await db.suppliers.update_one(
        {"_id": ObjectId(supplier_id)},
        {"$set": update_fields}
    )
    
    return {"message": "Supplier updated successfully"}


@app.delete("/api/suppliers/{supplier_id}")
async def delete_supplier(
    supplier_id: str,
    current_user: dict = Depends(require_role("specialist"))
):
    """Delete a supplier"""
    
    supplier = await db.suppliers.find_one({"_id": ObjectId(supplier_id)})
    
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    
    if supplier["created_by"] != str(current_user["_id"]) and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.suppliers.delete_one({"_id": ObjectId(supplier_id)})
    
    return {"message": "Supplier deleted successfully"}


@app.post("/api/suppliers/{supplier_id}/rate")
async def rate_supplier(
    supplier_id: str,
    rating_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Rate a supplier"""
    
    supplier = await db.suppliers.find_one({"_id": ObjectId(supplier_id)})
    
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    
    # Check if user already rated
    existing_review = await db.supplier_reviews.find_one({
        "supplier_id": supplier_id,
        "user_id": str(current_user["_id"])
    })
    
    review_doc = {
        "supplier_id": supplier_id,
        "user_id": str(current_user["_id"]),
        "user_name": current_user["name"],
        "rating": rating_data["rating"],
        "review": rating_data.get("review"),
        "created_at": datetime.utcnow()
    }
    
    if existing_review:
        await db.supplier_reviews.update_one(
            {"_id": existing_review["_id"]},
            {"$set": review_doc}
        )
    else:
        await db.supplier_reviews.insert_one(review_doc)
    
    # Recalculate average rating
    all_ratings = await db.supplier_reviews.find(
        {"supplier_id": supplier_id}
    ).to_list(1000)
    
    avg_rating = sum(r["rating"] for r in all_ratings) / len(all_ratings)
    
    await db.suppliers.update_one(
        {"_id": ObjectId(supplier_id)},
        {
            "$set": {
                "rating": round(avg_rating, 1),
                "total_ratings": len(all_ratings)
            }
        }
    )
    
    return {"message": "Rating submitted successfully"}


@app.get("/api/suppliers/products/list")
async def get_available_products(current_user: dict = Depends(get_current_user)):
    """Get list of all available products from suppliers"""
    
    pipeline = [
        {"$unwind": "$products_available"},
        {"$group": {"_id": "$products_available"}},
        {"$sort": {"_id": 1}}
    ]
    
    products = await db.suppliers.aggregate(pipeline).to_list(1000)
    
    return [p["_id"] for p in products]


@app.get("/api/suppliers/locations/list")
async def get_supplier_locations(current_user: dict = Depends(get_current_user)):
    """Get list of all supplier locations"""
    
    pipeline = [
        {"$group": {"_id": "$location"}},
        {"$sort": {"_id": 1}}
    ]
    
    locations = await db.suppliers.aggregate(pipeline).to_list(1000)
    
    return [l["_id"] for l in locations]

# Add this to your main.py file

async def seed_suppliers():
    """Seed initial suppliers data - Run this once to populate database"""
    
    suppliers_data = [
        {
            "name": "Green Valley Organics",
            "location": "Visakhapatnam, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43210",
                "email": "info@greenvalley.com",
                "address": "Plot 123, Agricultural Market, Visakhapatnam - 530001",
                "website": "www.greenvalley.com"
            },
            "products_available": ["Neem Oil", "Vermicompost", "Organic Seeds", "Bio-fertilizers", "Composting Materials", "Neem Cake"],
            "crop_association": ["Rice", "Cotton", "Vegetables", "Chili"],
            "rating": 4.5,
            "total_ratings": 45,
            "verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "name": "Nature Farm Supplies",
            "location": "Vizianagaram, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43211",
                "email": "contact@naturefarm.com",
                "address": "Main Road, Vizianagaram - 535002",
                "website": None
            },
            "products_available": ["Organic Fertilizers", "Bio-pesticides", "Neem Cake", "Panchagavya", "Jeevamrutham", "Amrutpani"],
            "crop_association": ["Groundnut", "Maize", "Sugarcane", "Pulses"],
            "rating": 4.3,
            "total_ratings": 32,
            "verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "name": "Tribal Agro Products",
            "location": "East Godavari, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43212",
                "email": "tribalagro@gmail.com",
                "address": "Tribal Area, East Godavari District - 533001",
                "website": None
            },
            "products_available": ["Traditional Seeds", "Organic Manure", "Herbal Pesticides", "Natural Growth Promoters", "Forest Honey"],
            "crop_association": ["Millets", "Pulses", "Traditional Crops", "Turmeric"],
            "rating": 4.8,
            "total_ratings": 56,
            "verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "name": "Eco Farming Solutions",
            "location": "Srikakulam, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43213",
                "email": "ecofarming@solutions.com",
                "address": "Agricultural Complex, Srikakulam - 532001",
                "website": "www.ecofarming.in"
            },
            "products_available": ["Neem Oil", "Composting Materials", "Bio-fertilizers", "Organic Seeds", "Growth Regulators", "Azospirillum"],
            "crop_association": ["Cashew", "Turmeric", "Vegetables", "Banana"],
            "rating": 4.2,
            "total_ratings": 28,
            "verified": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "name": "Organic Manure Hub",
            "location": "Guntur, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43214",
                "email": "organicmanurehub@gmail.com",
                "address": "Market Yard, Guntur - 522001",
                "website": None
            },
            "products_available": ["Vermicompost", "FYM", "Compost", "Green Manure Seeds", "Phospho-bacteria", "Azotobacter"],
            "crop_association": ["Chili", "Cotton", "Rice", "Tobacco"],
            "rating": 4.6,
            "total_ratings": 41,
            "verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "name": "Bio Agro Inputs",
            "location": "Krishna District, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43215",
                "email": "bioagro@inputs.com",
                "address": "Vijayawada - 520001",
                "website": "www.bioagroinputs.com"
            },
            "products_available": ["Trichoderma", "Pseudomonas", "Bio-fungicides", "Neem Products", "Pheromone Traps"],
            "crop_association": ["Paddy", "Sugarcane", "Banana", "Mango"],
            "rating": 4.4,
            "total_ratings": 38,
            "verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "name": "Green Earth Organics",
            "location": "Anantapur, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43216",
                "email": "greenearth@organic.com",
                "address": "Anantapur - 515001",
                "website": None
            },
            "products_available": ["Organic Pesticides", "Neem Oil", "Karanja Oil", "Fish Amino Acid", "Seaweed Extract"],
            "crop_association": ["Groundnut", "Tomato", "Ragi", "Pulses"],
            "rating": 4.1,
            "total_ratings": 24,
            "verified": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "name": "Farmers Pride Co-operative",
            "location": "Kurnool, Andhra Pradesh",
            "contact_info": {
                "phone": "+91 98765 43217",
                "email": "farmerspride@coop.com",
                "address": "Co-operative Building, Kurnool - 518001",
                "website": None
            },
            "products_available": ["Organic Seeds", "Bio-fertilizers", "Vermicompost", "Organic Pesticides", "Mulching Materials"],
            "crop_association": ["Cotton", "Jowar", "Maize", "Sunflower"],
            "rating": 4.7,
            "total_ratings": 52,
            "verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    try:
        # Check if suppliers already exist
        existing_count = await db.suppliers.count_documents({})
        if existing_count > 0:
            print(f"Suppliers already exist ({existing_count} records). Skipping seed.")
            return
        
        # Insert suppliers
        result = await db.suppliers.insert_many(suppliers_data)
        print(f"Successfully seeded {len(result.inserted_ids)} suppliers!")
        
    except Exception as e:
        print(f"Error seeding suppliers: {e}")


# Add an endpoint to trigger seeding (for development only)
@app.post("/api/admin/seed-suppliers")
async def trigger_seed_suppliers(current_user: dict = Depends(require_role("specialist"))):
    """Seed suppliers data - Admin/Specialist only"""
    await seed_suppliers()
    return {"message": "Suppliers seeded successfully"}

async def seed_video_tutorials():
    """Seed initial video tutorial data with YouTube videos"""
    
    specialist_user = await db.users.find_one({"role": "specialist"})
    created_by_id = str(specialist_user["_id"]) if specialist_user else "000000000000000000000000"
    
    videos_data = [
        # YouTube videos - Real farming education videos
        {
            "title": "Complete Organic Farming Guide for Beginners",
            "description": "Learn the basics of organic farming from scratch. This comprehensive guide covers soil preparation, organic fertilizers, pest control, and sustainable farming practices perfect for beginners.",
            "category": "soil_management",
            "duration_minutes": 25,
            "language": "english",
            "video_url": "https://www.youtube.com/watch?v=0h6QBivFZMo",
            "video_source": "youtube",
            "thumbnail_url": "https://img.youtube.com/vi/0h6QBivFZMo/maxresdefault.jpg",
            "topics_covered": ["Organic Farming", "Soil Preparation", "Sustainable Practices"],
            "difficulty_level": "beginner",
            "suitable_for_crops": ["All Crops"],
            "season_relevance": ["All Seasons"],
            "views_count": 3200,
            "likes_count": 245,
            "completion_rate": 0.82,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "Natural Pest Control Methods",
            "description": "Discover effective natural and organic methods to control pests without harmful chemicals. Learn about neem oil, companion planting, and biological pest control.",
            "category": "pest_control",
            "duration_minutes": 18,
            "language": "english",
            "video_url": "https://www.youtube.com/watch?v=eKPjznbRY3s",
            "video_source": "youtube",
            "thumbnail_url": "https://img.youtube.com/vi/eKPjznbRY3s/maxresdefault.jpg",
            "topics_covered": ["Pest Control", "Neem Oil", "Companion Planting"],
            "difficulty_level": "beginner",
            "suitable_for_crops": ["Vegetables", "Fruits"],
            "season_relevance": ["All Seasons"],
            "views_count": 2800,
            "likes_count": 198,
            "completion_rate": 0.75,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "Composting Guide - How to Make Compost",
            "description": "Step-by-step guide to creating nutrient-rich compost for your organic farm. Learn the perfect carbon to nitrogen ratio and composting techniques.",
            "category": "composting",
            "duration_minutes": 15,
            "language": "english",
            "video_url": "https://www.youtube.com/watch?v=NHr4ijG6GRg",
            "video_source": "youtube",
            "thumbnail_url": "https://img.youtube.com/vi/NHr4ijG6GRg/maxresdefault.jpg",
            "topics_covered": ["Composting", "Organic Matter", "Soil Health"],
            "difficulty_level": "beginner",
            "suitable_for_crops": ["All Crops"],
            "season_relevance": ["All Seasons"],
            "views_count": 1900,
            "likes_count": 156,
            "completion_rate": 0.78,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        # Sample uploaded videos
        {
            "title": "Neem Oil Preparation and Application",
            "description": "Learn how to prepare organic neem oil spray at home and apply it effectively to control pests in your crops. This traditional method is safe and effective.",
            "category": "pest_control",
            "duration_minutes": 12,
            "language": "telugu",
            "video_url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
            "video_source": "uploaded",
            "thumbnail_url": "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=640&h=360&fit=crop",
            "topics_covered": ["Neem Oil", "Organic Pesticide", "Pest Control"],
            "difficulty_level": "beginner",
            "suitable_for_crops": ["Cotton", "Vegetables", "Rice"],
            "season_relevance": ["Kharif", "Rabi"],
            "views_count": 1250,
            "likes_count": 98,
            "completion_rate": 0.85,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "Vermicompost Making at Home",
            "description": "Step-by-step guide to create high-quality vermicompost using kitchen waste and earthworms. Perfect for small-scale organic farming.",
            "category": "composting",
            "duration_minutes": 15,
            "language": "telugu",
            "video_url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
            "video_source": "uploaded",
            "thumbnail_url": "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=640&h=360&fit=crop",
            "topics_covered": ["Vermicompost", "Organic Fertilizer", "Waste Management"],
            "difficulty_level": "beginner",
            "suitable_for_crops": ["All Crops"],
            "season_relevance": ["All Seasons"],
            "views_count": 2100,
            "likes_count": 165,
            "completion_rate": 0.78,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "Drip Irrigation Setup Guide",
            "description": "Complete tutorial on setting up an efficient drip irrigation system for small farms. Save water and increase crop yields.",
            "category": "irrigation",
            "duration_minutes": 18,
            "language": "telugu",
            "video_url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
            "video_source": "uploaded",
            "thumbnail_url": "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=640&h=360&fit=crop",
            "topics_covered": ["Drip Irrigation", "Water Conservation", "Installation"],
            "difficulty_level": "intermediate",
            "suitable_for_crops": ["Vegetables", "Cotton", "Banana"],
            "season_relevance": ["Summer", "All Seasons"],
            "views_count": 890,
            "likes_count": 72,
            "completion_rate": 0.65,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "Panchagavya Preparation",
            "description": "Traditional organic growth promoter made from five cow products. Learn the complete preparation method step by step.",
            "category": "soil_management",
            "duration_minutes": 10,
            "language": "telugu",
            "video_url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
            "video_source": "uploaded",
            "thumbnail_url": "https://images.unsplash.com/photo-1500595046743-cd271d694d30?w=640&h=360&fit=crop",
            "topics_covered": ["Panchagavya", "Organic Growth Promoter", "Traditional Methods"],
            "difficulty_level": "beginner",
            "suitable_for_crops": ["All Crops"],
            "season_relevance": ["All Seasons"],
            "views_count": 1580,
            "likes_count": 142,
            "completion_rate": 0.88,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "title": "Seed Treatment with Trichoderma",
            "description": "Learn how to treat seeds with Trichoderma to prevent soil-borne diseases and improve germination rates.",
            "category": "seed_treatment",
            "duration_minutes": 8,
            "language": "telugu",
            "video_url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
            "video_source": "uploaded",
            "thumbnail_url": "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?w=640&h=360&fit=crop",
            "topics_covered": ["Trichoderma", "Seed Treatment", "Disease Prevention"],
            "difficulty_level": "intermediate",
            "suitable_for_crops": ["Cotton", "Rice", "Groundnut"],
            "season_relevance": ["Kharif", "Rabi"],
            "views_count": 745,
            "likes_count": 58,
            "completion_rate": 0.72,
            "created_by": created_by_id,
            "verified_by_specialist": True,
            "is_featured": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    try:
        existing_count = await db.video_tutorials.count_documents({})
        if existing_count > 0:
            print(f"Video tutorials already exist ({existing_count} records). Skipping seed.")
            return
        
        result = await db.video_tutorials.insert_many(videos_data)
        print(f"Successfully seeded {len(result.inserted_ids)} video tutorials!")
        
    except Exception as e:
        print(f"Error seeding video tutorials: {e}")
@app.post("/api/admin/seed-video-tutorials")
async def trigger_seed_video_tutorials(current_user: dict = Depends(require_role("specialist"))):
    """Seed video tutorials data - Specialist only"""
    await seed_video_tutorials()
    return {"message": "Video tutorials seeded successfully"}


# ============= Health Check =============

# ============= IMPACT DASHBOARD ROUTES =============

# ============= COMPREHENSIVE IMPACT DASHBOARD =============

@app.get("/api/impact/dashboard")
async def get_impact_dashboard(
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive impact dashboard based on all platform activities"""
    
    user_id = str(current_user["_id"])
    
    # ========== CORE STATISTICS ==========
    
    # Crop Analysis Impact
    total_analyses = await db.crop_photos.count_documents({"user_id": user_id})
    successful_treatments = await db.treatment_submissions.count_documents({
        "user_id": user_id,
        "outcome": "success"
    })
    total_treatments = await db.treatment_submissions.count_documents({"user_id": user_id})
    
    success_rate = (successful_treatments / total_treatments * 100) if total_treatments > 0 else 0
    
    # Organic Solutions Impact
    organic_solutions_applied = await db.solution_applications.count_documents({
        "user_id": user_id,
        "status": {"$in": ["applied", "completed"]}
    })
    
    completed_solutions = await db.solution_applications.count_documents({
        "user_id": user_id,
        "status": "completed",
        "outcome": "success"
    })
    
    # Traditional Practices Impact
    traditional_practices_used = await db.practice_applications.count_documents({
        "user_id": user_id
    })
    
    # Video Learning Impact
    videos_watched = await db.video_progress.count_documents({
        "user_id": user_id,
        "completed": True
    })
    
    # Community Impact
    posts_created = await db.community_posts.count_documents({"author_id": user_id})
    comments_made = await db.community_comments.count_documents({"user_id": user_id})
    helpful_posts = await db.community_posts.count_documents({
        "author_id": user_id,
        "helpful_count": {"$gt": 0}
    })
    
    # Consultation Impact (if farmer helped others)
    consultations_received = await db.consultation_sessions.count_documents({
        "farmer_id": user_id,
        "status": "completed"
    })
    
    # ========== COST SAVINGS CALCULATION ==========
    
    # Chemical pesticide cost vs organic: ₹500/acre vs ₹150/acre (saving ₹350)
    pesticide_savings = organic_solutions_applied * 350
    
    # Fertilizer cost: Chemical ₹3000/acre vs Organic ₹1200/acre (saving ₹1800)
    fertilizer_savings = traditional_practices_used * 1800
    
    # Labor cost reduction through knowledge: ₹200 per successful treatment
    labor_savings = successful_treatments * 200
    
    # Prevented crop loss (early detection): ₹5000 per crop saved
    crop_loss_prevention = total_analyses * 5000 * (success_rate / 100)
    
    total_cost_saved = pesticide_savings + fertilizer_savings + labor_savings + crop_loss_prevention
    
    cost_breakdown = {
        "organic_pesticides_savings": round(pesticide_savings, 2),
        "organic_fertilizers_savings": round(fertilizer_savings, 2),
        "labor_cost_reduction": round(labor_savings, 2),
        "crop_loss_prevention": round(crop_loss_prevention, 2)
    }
    
    # ========== ENVIRONMENTAL IMPACT ==========
    
    # Chemical reduction (2.5 kg per organic solution applied)
    chemicals_reduced_kg = organic_solutions_applied * 2.5
    chemical_reduction_percentage = min(organic_solutions_applied * 8, 100)
    
    # Water conservation (500L per drip irrigation/mulching practice)
    water_saved_liters = traditional_practices_used * 500
    
    # Carbon footprint (15kg CO2 per organic treatment vs chemical)
    carbon_reduced_kg = (organic_solutions_applied + traditional_practices_used) * 15
    
    # Soil health improvement (organic matter increase)
    soil_health_score = min((organic_solutions_applied + traditional_practices_used) * 5, 100)
    
    # ========== MONTHLY PROGRESS (Last 30 days) ==========
    
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    recent_solutions = await db.solution_applications.count_documents({
        "user_id": user_id,
        "applied_at": {"$gte": thirty_days_ago}
    })
    
    recent_analyses = await db.crop_photos.count_documents({
        "user_id": user_id,
        "uploaded_at": {"$gte": thirty_days_ago}
    })
    
    recent_community = await db.community_posts.count_documents({
        "author_id": user_id,
        "created_at": {"$gte": thirty_days_ago}
    }) + await db.community_comments.count_documents({
        "user_id": user_id,
        "created_at": {"$gte": thirty_days_ago}
    })
    
    recent_videos = await db.video_progress.count_documents({
        "user_id": user_id,
        "completed": True,
        "completed_at": {"$gte": thirty_days_ago}
    })
    
    # Progress percentages
    organic_solutions_progress = min((recent_solutions / 10) * 100, 100)
    crop_health_progress = min((recent_analyses / 8) * 100, 100)
    community_progress = min((recent_community / 5) * 100, 100)
    learning_progress = min((recent_videos / 3) * 100, 100)
    
    # ========== ACHIEVEMENTS ==========
    
    achievements = []
    
    # Analysis achievements
    if total_analyses >= 50:
        achievements.append({
            "title": "Master Analyst",
            "description": "Analyzed 50+ crop photos",
            "icon": "🔬",
            "category": "analysis",
            "earned_at": datetime.utcnow().isoformat()
        })
    elif total_analyses >= 20:
        achievements.append({
            "title": "Expert Diagnostician",
            "description": "Analyzed 20+ crop photos",
            "icon": "🔍",
            "category": "analysis",
            "earned_at": datetime.utcnow().isoformat()
        })
    elif total_analyses >= 10:
        achievements.append({
            "title": "Crop Monitor",
            "description": "Analyzed 10+ crop photos",
            "icon": "🌾",
            "category": "analysis",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # Success achievements
    if success_rate >= 90:
        achievements.append({
            "title": "Success Champion",
            "description": "90%+ treatment success rate",
            "icon": "🏆",
            "category": "success",
            "earned_at": datetime.utcnow().isoformat()
        })
    elif success_rate >= 75:
        achievements.append({
            "title": "Treatment Expert",
            "description": "75%+ treatment success rate",
            "icon": "⭐",
            "category": "success",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # Organic achievements
    if organic_solutions_applied >= 20:
        achievements.append({
            "title": "Organic Warrior",
            "description": "Applied 20+ organic solutions",
            "icon": "🌱",
            "category": "organic",
            "earned_at": datetime.utcnow().isoformat()
        })
    elif organic_solutions_applied >= 10:
        achievements.append({
            "title": "Organic Advocate",
            "description": "Applied 10+ organic solutions",
            "icon": "🍃",
            "category": "organic",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # Traditional knowledge achievements
    if traditional_practices_used >= 10:
        achievements.append({
            "title": "Tradition Keeper",
            "description": "Used 10+ traditional practices",
            "icon": "📜",
            "category": "traditional",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # Community achievements
    if posts_created >= 10:
        achievements.append({
            "title": "Community Leader",
            "description": "Created 10+ helpful posts",
            "icon": "👥",
            "category": "community",
            "earned_at": datetime.utcnow().isoformat()
        })
    elif posts_created >= 5:
        achievements.append({
            "title": "Active Contributor",
            "description": "Created 5+ posts",
            "icon": "💬",
            "category": "community",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # Learning achievements
    if videos_watched >= 10:
        achievements.append({
            "title": "Knowledge Seeker",
            "description": "Completed 10+ video tutorials",
            "icon": "📚",
            "category": "learning",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # Environmental achievements
    if chemicals_reduced_kg >= 50:
        achievements.append({
            "title": "Eco Warrior",
            "description": f"Reduced {chemicals_reduced_kg}kg chemicals",
            "icon": "🌍",
            "category": "environment",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # Cost saving achievements
    if total_cost_saved >= 10000:
        achievements.append({
            "title": "Smart Farmer",
            "description": f"Saved ₹{int(total_cost_saved):,}",
            "icon": "💰",
            "category": "savings",
            "earned_at": datetime.utcnow().isoformat()
        })
    
    # ========== RESPONSE ==========
    
    return {
        "stats": {
            "total_crops_monitored": total_analyses,
            "treatments_applied": total_treatments,
            "successful_treatments": successful_treatments,
            "success_rate": round(success_rate, 1),
            "cost_saved": round(total_cost_saved, 2),
            "chemical_reduction": round(chemical_reduction_percentage, 1),
            "organic_solutions_adopted": organic_solutions_applied,
            "traditional_practices_used": traditional_practices_used,
            "videos_completed": videos_watched,
            "community_contributions": posts_created + comments_made,
            "consultations_received": consultations_received,
            "badges_earned": len(achievements),
            "streak_days": current_user.get("streak_count", 0)
        },
        "monthly_progress": {
            "organic_solutions": {
                "progress": round(organic_solutions_progress, 1),
                "current": recent_solutions,
                "target": 10
            },
            "crop_health_monitoring": {
                "progress": round(crop_health_progress, 1),
                "current": recent_analyses,
                "target": 8
            },
            "community_engagement": {
                "progress": round(community_progress, 1),
                "current": recent_community,
                "target": 5
            },
            "learning_sessions": {
                "progress": round(learning_progress, 1),
                "current": recent_videos,
                "target": 3
            }
        },
        "cost_savings": {
            "total_saved": round(total_cost_saved, 2),
            "breakdown": cost_breakdown
        },
        "environmental_impact": {
            "chemicals_reduced_kg": round(chemicals_reduced_kg, 1),
            "chemical_reduction_percentage": round(chemical_reduction_percentage, 1),
            "water_saved_liters": round(water_saved_liters, 1),
            "carbon_reduced_kg": round(carbon_reduced_kg, 1),
            "soil_health_score": round(soil_health_score, 1)
        },
        "achievements": achievements,
        "user_info": {
            "name": current_user["name"],
            "role": current_user.get("role", "farmer"),
            "location": current_user.get("village"),
            "district": current_user.get("district"),
            "member_since": current_user["created_at"].strftime("%B %Y")
        }
    }

@app.get("/api/impact/timeline")
async def get_impact_timeline(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get timeline of user's activities and achievements"""
    
    user_id = str(current_user["_id"])
    start_date = datetime.utcnow() - timedelta(days=days)
    
    timeline = []
    
    # Get crop analyses
    photos = await db.crop_photos.find({
        "user_id": user_id,
        "uploaded_at": {"$gte": start_date}
    }).sort("uploaded_at", -1).to_list(50)
    
    for photo in photos:
        timeline.append({
            "type": "crop_analysis",
            "title": f"Analyzed {photo.get('disease', 'crop issue')}",
            "description": f"Detected with {int(photo.get('confidence_score', 0)*100)}% confidence",
            "icon": "🔬",
            "date": photo["uploaded_at"].isoformat()
        })
    
    # Get solution applications
    solutions = await db.solution_applications.find({
        "user_id": user_id,
        "applied_at": {"$gte": start_date}
    }).sort("applied_at", -1).to_list(50)
    
    for solution in solutions:
        sol_data = await db.organic_solutions.find_one({"_id": ObjectId(solution["solution_id"])})
        if sol_data:
            timeline.append({
                "type": "solution_applied",
                "title": f"Applied {sol_data['title']}",
                "description": f"Status: {solution['status']}",
                "icon": "🌱",
                "date": solution["applied_at"].isoformat()
            })
    
    # Get community posts
    posts = await db.community_posts.find({
        "author_id": user_id,
        "created_at": {"$gte": start_date}
    }).sort("created_at", -1).to_list(50)
    
    for post in posts:
        timeline.append({
            "type": "community_post",
            "title": f"Posted: {post['title'][:50]}...",
            "description": f"{len(post.get('comments', []))} comments",
            "icon": "💬",
            "date": post["created_at"].isoformat()
        })
    
    # Sort by date
    timeline.sort(key=lambda x: x["date"], reverse=True)
    
    return timeline[:20]

# voice translation endpoint
# ============= VOICE & TRANSLATION ENDPOINTS =============

# Initialize Stripe

stripe.api_key = settings.STRIPE_SECRET_KEY
if stripe.api_key:
    print(f"✅ Stripe configured: {stripe.api_key[:20]}...")
else:
    print("⚠️ Stripe key not set!")
# ============= PRODUCT MARKETPLACE ROUTES =============

@app.post("/api/products")
async def create_product(
    product_data: ProductCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new product listing"""
    
    product_doc = {
        "seller_id": str(current_user["_id"]),
        "seller_name": current_user["name"],
        "title": product_data.title,
        "description": product_data.description,
        "category": product_data.category,
        "subcategory": product_data.subcategory,
        "price": product_data.price,
        "unit": product_data.unit,
        "min_order_quantity": product_data.min_order_quantity,
        "stock_available": product_data.stock_available,
        "stock_unit": product_data.stock_unit,
        "organic_certified": product_data.organic_certified,
        "brand": product_data.brand,
        "specifications": product_data.specifications,
        "images": product_data.images,
        "thumbnail": product_data.images[0] if product_data.images else None,
        "location": product_data.location,
        "district": product_data.district or current_user.get("district", ""),
        "state": "Andhra Pradesh",
        "suitable_for_crops": product_data.suitable_for_crops,
        "status": "active",
        "is_featured": False,
        "views_count": 0,
        "inquiries_count": 0,
        "orders_count": 0,
        "rating": 0.0,
        "total_ratings": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.products.insert_one(product_doc)
    
    return {
        "product_id": str(result.inserted_id),
        "message": "Product listed successfully"
    }


@app.get("/api/products")
async def get_products(
    category: Optional[str] = None,
    search: Optional[str] = None,
    district: Optional[str] = None,
    organic_only: bool = False,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get products with filters"""
    
    query = {"status": "active"}
    
    if category:
        query["category"] = category
    
    if search:
        query["$or"] = [
            {"title": {"$regex": search, "$options": "i"}},
            {"description": {"$regex": search, "$options": "i"}},
            {"brand": {"$regex": search, "$options": "i"}}
        ]
    
    if district:
        query["district"] = {"$regex": district, "$options": "i"}
    
    if organic_only:
        query["organic_certified"] = True
    
    if min_price is not None:
        query["price"] = {"$gte": min_price}
    if max_price is not None:
        if "price" not in query:
            query["price"] = {}
        query["price"]["$lte"] = max_price
    
    total = await db.products.count_documents(query)
    
    sort_direction = -1 if sort_order == "desc" else 1
    skip = (page - 1) * page_size
    
    products = await db.products.find(query).sort(
        sort_by, sort_direction
    ).skip(skip).limit(page_size).to_list(page_size)
    
    result = []
    for p in products:
        result.append({
            "id": str(p["_id"]),
            "seller_id": p["seller_id"],
            "seller_name": p["seller_name"],
            "title": p["title"],
            "description": p["description"][:200],
            "category": p["category"],
            "price": p["price"],
            "unit": p["unit"],
            "stock_available": p["stock_available"],
            "organic_certified": p.get("organic_certified", False),
            "thumbnail": p.get("thumbnail"),
            "location": p["location"],
            "district": p["district"],
            "rating": p.get("rating", 0),
            "orders_count": p.get("orders_count", 0),
            "created_at": p["created_at"].isoformat()
        })
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "products": result
    }

@app.get("/api/products/categories")
async def get_product_categories(current_user: dict = Depends(get_current_user)):
    """Get product categories"""
    return {
        "categories": [
            {"value": "seeds", "label": "Seeds", "icon": "🌱"},
            {"value": "fertilizers", "label": "Fertilizers", "icon": "🌿"},
            {"value": "pesticides", "label": "Pesticides", "icon": "🛡️"},
            {"value": "tools", "label": "Tools", "icon": "🔧"},
            {"value": "equipment", "label": "Equipment", "icon": "⚙️"},
            {"value": "produce", "label": "Fresh Produce", "icon": "🥬"}
        ]
    }


@app.get("/api/products/{product_id}")
async def get_product_detail(
    product_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get product details"""
    
    product = await db.products.find_one({"_id": ObjectId(product_id)})
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Increment views
    await db.products.update_one(
        {"_id": ObjectId(product_id)},
        {"$inc": {"views_count": 1}}
    )
    
    # Get seller info
    seller = await db.users.find_one({"_id": ObjectId(product["seller_id"])})
    
    # Get reviews
    reviews = await db.product_reviews.find(
        {"product_id": product_id}
    ).sort("created_at", -1).limit(10).to_list(10)
    
    return {
        "id": str(product["_id"]),
        "seller": {
            "id": product["seller_id"],
            "name": product["seller_name"],
            "phone": seller.get("phone") if seller else None,
            "district": seller.get("district") if seller else None
        },
        "title": product["title"],
        "description": product["description"],
        "category": product["category"],
        "subcategory": product.get("subcategory"),
        "price": product["price"],
        "unit": product["unit"],
        "min_order_quantity": product.get("min_order_quantity", 1),
        "stock_available": product["stock_available"],
        "stock_unit": product["stock_unit"],
        "organic_certified": product.get("organic_certified", False),
        "brand": product.get("brand"),
        "specifications": product.get("specifications", {}),
        "images": product.get("images", []),
        "location": product["location"],
        "district": product["district"],
        "suitable_for_crops": product.get("suitable_for_crops", []),
        "rating": product.get("rating", 0),
        "total_ratings": product.get("total_ratings", 0),
        "views_count": product.get("views_count", 0),
        "orders_count": product.get("orders_count", 0),
        "created_at": product["created_at"].isoformat(),
        "reviews": [
            {
                "user_name": r["user_name"],
                "rating": r["rating"],
                "review": r.get("review"),
                "verified_purchase": r.get("verified_purchase", False),
                "created_at": r["created_at"].isoformat()
            } for r in reviews
        ]
    }


# @app.post("/api/products/{product_id}/order")
# async def create_order(
#     product_id: str,
#     order_data: OrderCreateRequest,
#     current_user: dict = Depends(get_current_user)
# ):
#     """Create order and initialize Stripe payment"""
    
#     product = await db.products.find_one({"_id": ObjectId(product_id)})
    
#     if not product:
#         raise HTTPException(status_code=404, detail="Product not found")
    
#     if product["status"] != "active":
#         raise HTTPException(status_code=400, detail="Product not available")
    
#     if order_data.quantity > product["stock_available"]:
#         raise HTTPException(status_code=400, detail="Insufficient stock")
    
#     # Calculate total
#     total_price = product["price"] * order_data.quantity
    
#     # Generate order number
#     order_number = f"ORD{datetime.utcnow().strftime('%Y%m%d')}{str(uuid.uuid4())[:8].upper()}"
    
#     # Create order
#     order_doc = {
#         "order_number": order_number,
#         "buyer_id": str(current_user["_id"]),
#         "buyer_name": current_user["name"],
#         "buyer_phone": current_user.get("phone"),
#         "buyer_email": current_user.get("email"),
#         "seller_id": product["seller_id"],
#         "seller_name": product["seller_name"],
#         "product_id": product_id,
#         "product_title": product["title"],
#         "product_image": product.get("thumbnail"),
#         "quantity": order_data.quantity,
#         "unit": product["unit"],
#         "unit_price": product["price"],
#         "total_price": total_price,
#         "shipping_address": order_data.shipping_address,
#         "payment_status": "pending",
#         "payment_method": "stripe",
#         "order_status": "pending",
#         "buyer_notes": order_data.buyer_notes,
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow()
#     }
    
#     result = await db.product_orders.insert_one(order_doc)
#     order_id = str(result.inserted_id)
    
#     # Create Stripe Payment Intent
#     try:
#         payment_intent = stripe.PaymentIntent.create(
#             amount=int(total_price * 100),  # Amount in paise
#             currency="inr",
#             metadata={
#                 "order_id": order_id,
#                 "order_number": order_number,
#                 "product_title": product["title"]
#             }
#         )
        
#         # Update order with payment intent
#         await db.product_orders.update_one(
#             {"_id": ObjectId(order_id)},
#             {"$set": {"stripe_payment_intent_id": payment_intent.id}}
#         )
        
#         return {
#             "order_id": order_id,
#             "order_number": order_number,
#             "client_secret": payment_intent.client_secret,
#             "total_price": total_price,
#             "message": "Order created successfully"
#         }
        
#     except Exception as e:
#         # Delete order if payment intent creation fails
#         await db.product_orders.delete_one({"_id": ObjectId(order_id)})
#         raise HTTPException(status_code=500, detail=f"Payment initialization failed: {str(e)}")

@app.post("/api/products/{product_id}/order")
async def create_order(
    product_id: str,
    order_data: dict,  # Changed from OrderCreateRequest to dict
    current_user: dict = Depends(get_current_user)
):
    """Create order and initialize Stripe payment"""
    
    product = await db.products.find_one({"_id": ObjectId(product_id)})
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    if product["status"] != "active":
        raise HTTPException(status_code=400, detail="Product not available")
    
    # Extract and validate data
    quantity = order_data.get("quantity")
    shipping_address = order_data.get("shipping_address")
    buyer_notes = order_data.get("buyer_notes", "")
    
    if not quantity or not shipping_address:
        raise HTTPException(
            status_code=400, 
            detail="Quantity and shipping address are required"
        )
    
    if quantity > product["stock_available"]:
        raise HTTPException(status_code=400, detail="Insufficient stock")
    
    # Calculate total
    total_price = product["price"] * quantity
    
    # Generate order number
    order_number = f"ORD{datetime.utcnow().strftime('%Y%m%d')}{str(uuid.uuid4())[:8].upper()}"
    
    # Create order
    order_doc = {
        "order_number": order_number,
        "buyer_id": str(current_user["_id"]),
        "buyer_name": current_user["name"],
        "buyer_phone": current_user.get("phone"),
        "buyer_email": current_user.get("email"),
        "seller_id": product["seller_id"],
        "seller_name": product["seller_name"],
        "product_id": product_id,
        "product_title": product["title"],
        "product_image": product.get("thumbnail"),
        "quantity": quantity,
        "unit": product["unit"],
        "unit_price": product["price"],
        "total_price": total_price,
        "shipping_address": shipping_address,
        "payment_status": "pending",
        "payment_method": "stripe",
        "order_status": "pending",
        "buyer_notes": buyer_notes,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.product_orders.insert_one(order_doc)
    order_id = str(result.inserted_id)
    
    # Create Stripe Payment Intent
    try:
        payment_intent = stripe.PaymentIntent.create(
            amount=int(total_price * 100),  # Amount in paise
            currency="inr",
            metadata={
                "order_id": order_id,
                "order_number": order_number,
                "product_title": product["title"]
            }
        )
        
        # Update order with payment intent
        await db.product_orders.update_one(
            {"_id": ObjectId(order_id)},
            {"$set": {"stripe_payment_intent_id": payment_intent.id}}
        )
        
        return {
            "order_id": order_id,
            "order_number": order_number,
            "client_secret": payment_intent.client_secret,
            "total_price": total_price,
            "message": "Order created successfully"
        }
        
    except Exception as e:
        # Delete order if payment intent creation fails
        await db.product_orders.delete_one({"_id": ObjectId(order_id)})
        raise HTTPException(status_code=500, detail=f"Payment initialization failed: {str(e)}")
    
@app.post("/api/products/orders/{order_id}/confirm-payment")
async def confirm_payment(
    order_id: str,
    payment_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Confirm payment after Stripe success"""
    
    order = await db.product_orders.find_one({"_id": ObjectId(order_id)})
    
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    if order["buyer_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Verify payment with Stripe
    try:
        payment_intent = stripe.PaymentIntent.retrieve(order["stripe_payment_intent_id"])
        
        if payment_intent.status == "succeeded":
            # Get charge ID safely
            charge_id = None
            if hasattr(payment_intent, 'charges') and payment_intent.charges and hasattr(payment_intent.charges, 'data') and len(payment_intent.charges.data) > 0:
                charge_id = payment_intent.charges.data[0].id
            elif hasattr(payment_intent, 'latest_charge') and payment_intent.latest_charge:
                charge_id = payment_intent.latest_charge
            
            # Update order
            await db.product_orders.update_one(
                {"_id": ObjectId(order_id)},
                {
                    "$set": {
                        "payment_status": "completed",
                        "payment_completed_at": datetime.utcnow(),
                        "order_status": "confirmed",
                        "stripe_charge_id": charge_id,
                        "stripe_payment_intent_id": payment_intent.id,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Payment verification failed: {str(e)}")


@app.get("/api/products/orders/my-orders")
async def get_my_orders(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get user's orders (as buyer)"""
    
    query = {"buyer_id": str(current_user["_id"])}
    
    if status:
        query["order_status"] = status
    
    orders = await db.product_orders.find(query).sort(
        "created_at", -1
    ).to_list(100)
    
    return [
        {
            "id": str(o["_id"]),
            "order_number": o["order_number"],
            "product_title": o["product_title"],
            "product_image": o.get("product_image"),
            "seller_name": o["seller_name"],
            "quantity": o["quantity"],
            "unit": o["unit"],
            "total_price": o["total_price"],
            "order_status": o["order_status"],
            "payment_status": o["payment_status"],
            "created_at": o["created_at"].isoformat(),
            "estimated_delivery": o.get("estimated_delivery").isoformat() if o.get("estimated_delivery") else None
        } for o in orders
    ]


@app.get("/api/products/orders/my-sales")
async def get_my_sales(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get user's orders (as seller)"""
    
    query = {"seller_id": str(current_user["_id"])}
    
    if status:
        query["order_status"] = status
    
    orders = await db.product_orders.find(query).sort(
        "created_at", -1
    ).to_list(100)
    
    return [
        {
            "id": str(o["_id"]),
            "order_number": o["order_number"],
            "product_title": o["product_title"],
            "buyer_name": o["buyer_name"],
            "buyer_phone": o.get("buyer_phone"),
            "buyer_address": o.get("shipping_address"),
            "quantity": o["quantity"],
            "unit": o["unit"],
            "total_price": o["total_price"],
            "order_status": o["order_status"],
            "payment_status": o["payment_status"],
            "created_at": o["created_at"].isoformat()
        } for o in orders
    ]


@app.get("/api/products/my-products")
async def get_my_products(
    current_user: dict = Depends(get_current_user)
):
    """Get user's product listings"""
    
    products = await db.products.find(
        {"seller_id": str(current_user["_id"])}
    ).sort("created_at", -1).to_list(100)
    
    return [
        {
            "id": str(p["_id"]),
            "title": p["title"],
            "category": p["category"],
            "price": p["price"],
            "unit": p["unit"],
            "stock_available": p["stock_available"],
            "status": p["status"],
            "views_count": p.get("views_count", 0),
            "orders_count": p.get("orders_count", 0),
            "thumbnail": p.get("thumbnail"),
            "created_at": p["created_at"].isoformat()
        } for p in products
    ]


@app.put("/api/products/{product_id}")
async def update_product(
    product_id: str,
    update_data: ProductUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update product"""
    
    product = await db.products.find_one({"_id": ObjectId(product_id)})
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    if product["seller_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_fields = update_data.dict(exclude_unset=True)
    update_fields["updated_at"] = datetime.utcnow()
    
    await db.products.update_one(
        {"_id": ObjectId(product_id)},
        {"$set": update_fields}
    )
    
    return {"message": "Product updated successfully"}


@app.delete("/api/products/{product_id}")
async def delete_product(
    product_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete product"""
    
    product = await db.products.find_one({"_id": ObjectId(product_id)})
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    if product["seller_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Check for pending orders
    pending_orders = await db.product_orders.count_documents({
        "product_id": product_id,
        "order_status": {"$in": ["pending", "confirmed", "preparing", "shipped"]}
    })
    
    if pending_orders > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete product with {pending_orders} pending order(s)"
        )
    
    await db.products.update_one(
        {"_id": ObjectId(product_id)},
        {"$set": {"status": "archived", "updated_at": datetime.utcnow()}}
    )
    
    return {"message": "Product archived successfully"}


# ============= SEED PRODUCTS DATA =============
async def seed_products():
    """Seed sample products in Telugu"""
    
    # Get some users to assign as sellers
    farmers = await db.users.find({"role": "farmer"}).limit(10).to_list(10)
    
    if not farmers:
        print("No farmers found. Please create users first.")
        return
    
    products_data = [
        # Seeds
        {
            "seller_id": str(farmers[0]["_id"]),
            "seller_name": farmers[0]["name"],
            "title": "సేంద్రీయ టమోటా గింజలు - హైబ్రిడ్ వెరైటీ",
            "description": "ఎక్కువ ఫలితాలు ఇచ్చే సేంద్రీయ టమోటా గింజలు. ఖరీఫ్ మరియు రాబీ సీజన్లకు అనుకూలం. రోగ నిరోధక వెరైటీ, రుచి మరియు నిల్వకాలం మెరుగ్గా ఉంది.",
            "category": "seeds",
            "subcategory": "vegetable_seeds",
            "price": 150.0,
            "unit": "packet (100g)",
            "min_order_quantity": 1,
            "stock_available": 50,
            "stock_unit": "packets",
            "organic_certified": True,
            "brand": "Green Valley Seeds",
            "specifications": {
                "germination_rate": "85%",
                "days_to_maturity": "60-70 రోజులు",
                "plant_height": "120-150cm"
            },
            "images": ["https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=800"],
            "location": "Visakhapatnam",
            "district": "Visakhapatnam",
            "suitable_for_crops": ["టమోటా"],
            "status": "active",
            "created_at": datetime.utcnow()
        },
        # Fertilizers
        {
            "seller_id": str(farmers[1]["_id"]),
            "seller_name": farmers[1]["name"],
            "title": "ప్రీమియం వర్మికంపోస్ట్ - 100% సేంద్రీయ",
            "description": "భూకీడల నుండి తయారు చేసిన ప్రీమియం వర్మికంపోస్ట్. NPK మరియు సూక్ష్మపోషకాలలో సమృద్ధిగా ఉంటుంది. అన్ని రకాల పంటలకు సరైనది. నేల ఆరోగ్యాన్ని సహజంగా మెరుగుపరుస్తుంది.",
            "category": "fertilizers",
            "subcategory": "organic_fertilizer",
            "price": 300.0,
            "unit": "per 10kg bag",
            "min_order_quantity": 1,
            "stock_available": 100,
            "stock_unit": "bags",
            "organic_certified": True,
            "brand": "Nature's Best",
            "specifications": {
                "npk_ratio": "2:1:1",
                "moisture_content": "35-40%",
                "ph_level": "6.5-7.5"
            },
            "images": ["https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=800"],
            "location": "Guntur",
            "district": "Guntur",
            "suitable_for_crops": ["అన్ని పంటలు"],
            "status": "active",
            "created_at": datetime.utcnow()
        },
        # Tools
        {
            "seller_id": str(farmers[2]["_id"]),
            "seller_name": farmers[2]["name"],
            "title": "గార్డెన్ హ్యాండ్ టూల్స్ కిట్ - 5 భాగాలు",
            "description": "అవసరమైన తోట పరికరాల సెట్, అందులో ట్రావెల్, వీడర్, కల్టివేటర్, ఫోర్క్, ప్రూనర్ ఉన్నాయి. అధిక-నాణ్యత గల స్టీల్ తో తయారు. సౌకర్యవంతమైన చెక్క హ్యాండిల్.",
            "category": "tools",
            "subcategory": "hand_tools",
            "price": 450.0,
            "unit": "per set",
            "min_order_quantity": 1,
            "stock_available": 20,
            "stock_unit": "sets",
            "organic_certified": False,
            "brand": "FarmPro",
            "specifications": {
                "material": "స్టెయిన్లెస్ స్టీల్",
                "handle": "చెక్క",
                "warranty": "1 సంవత్సరం"
            },
            "images": ["https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=800"],
            "location": "Vijayawada",
            "district": "Krishna",
            "suitable_for_crops": ["అన్ని పంటలు"],
            "status": "active",
            "created_at": datetime.utcnow()
        },
        # Organic Pesticides
        {
            "seller_id": str(farmers[3]["_id"]),
            "seller_name": farmers[3]["name"],
            "title": "నీమ్ ఆయిల్ కాంక్ట్రేట్ - సేంద్రీయ పీestsicide",
            "description": "సేంద్రీయ కీటక నియంత్రణ కోసం స్వచ్ఛమైన నీమ్ ఆయిల్. ఆఫిడ్స్, వైట్‌ఫ్లీస్, మరియు మైట్స్ పై ప్రభావవంతం. లాభకరమైన కీటకాలకు సురక్షితం. ఉపయోగించడానికి ముందే నీటితో కలపండి.",
            "category": "pesticides",
            "subcategory": "organic_pesticide",
            "price": 250.0,
            "unit": "per liter",
            "min_order_quantity": 1,
            "stock_available": 40,
            "stock_unit": "liters",
            "organic_certified": True,
            "brand": "BioDefend",
            "specifications": {
                "azadirachtin_content": "1500ppm",
                "dilution_ratio": "1:200",
                "shelf_life": "2 సంవత్సరాలు"
            },
            "images": ["https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=800"],
            "location": "Anantapur",
            "district": "Anantapur",
            "suitable_for_crops": ["కాటన్", "కూరగాయలు", "బియ్యం"],
            "status": "active",
            "created_at": datetime.utcnow()
        }
        # You can continue translating the remaining products in the same way...
    ]
    
    try:
        existing_count = await db.products.count_documents({})
        if existing_count > 0:
            print(f"Products already exist ({existing_count} records). Skipping seed.")
            return
        
        result = await db.products.insert_many(products_data)
        print(f"Successfully seeded {len(result.inserted_ids)} products!")
        
    except Exception as e:
        print(f"Error seeding products: {e}")


@app.post("/api/admin/seed-products")
async def trigger_seed_products(current_user: dict = Depends(get_current_user)):
    """Seed products data"""
    await seed_products()
    return {"message": "Products seeded successfully"}



# chat bot intergration 

# Add these imports to your main.py
import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# ============= CHATBOT KNOWLEDGE BASE =============

class ChatbotKnowledgeBase:
    def __init__(self):
        self.intents = {
            "greeting": {
                "patterns": [
                    "hello", "hi", "hey", "good morning", "good evening", 
                    "namaste", "how are you", "what's up", "greetings"
                ],
                "responses": [
                    "Hello! Welcome to Organic Advisory System. How can I help you today?",
                    "Hi there! I'm here to assist you with organic farming. What would you like to know?",
                    "Namaste! How can I help you with your crops today?",
                    "Hello! I'm your farming assistant. Ask me about diseases, treatments, or farming tips!"
                ],
                "context": ["greeting"]
            },
            "disease_inquiry": {
                "patterns": [
                    "disease", "sick plant", "plant problem", "crop issue", 
                    "leaf spot", "blight", "rot", "fungal infection", 
                    "pest attack", "yellowing leaves", "wilting", "dying crop"
                ],
                "responses": [
                    "I can help you identify crop diseases! To get accurate diagnosis:\n\n1. Go to 'Crop Analysis' section\n2. Upload a clear photo of the affected plant\n3. Our AI will detect the disease with confidence score\n4. You'll get organic treatment recommendations\n\nWould you like me to show you recent disease reports in your area?",
                    "For disease identification, please upload a photo in the Crop Analysis section. Make sure the photo is:\n- Well-lit and clear\n- Focused on affected areas\n- Shows leaves, stems, or fruits\n\nOur system can detect 50+ common crop diseases instantly!"
                ],
                "context": ["disease", "diagnosis"],
                "follow_up": ["treatment_inquiry", "upload_photo"]
            },
            "treatment_inquiry": {
                "patterns": [
                    "treatment", "cure", "solution", "remedy", "how to fix",
                    "organic treatment", "pesticide", "control", "manage disease"
                ],
                "responses": [
                    "We have 100+ proven organic treatment solutions! Here's what you can explore:\n\n📚 Organic Solutions Library\n- Neem oil preparations\n- Vermicompost recipes\n- Natural pesticides\n- Success rate: 85%+\n\n🌿 Traditional Practices\n- Tribal farming wisdom\n- Verified by elders\n- Time-tested methods\n\n🎥 Video Tutorials\n- Step-by-step guides\n- In Telugu & English\n\nWhich would you like to explore first?",
                    "For organic treatments, I recommend:\n\n1. **Neem Oil Spray** - ₹150/acre\n   - Effective against aphids, whiteflies\n   - 85% success rate\n\n2. **Panchagavya** - ₹100/acre\n   - Growth promoter\n   - 90% success rate\n\n3. **Vermicompost** - ₹300/10kg\n   - Soil health improver\n   - Rich in NPK\n\nWant detailed recipes for any of these?"
                ],
                "context": ["treatment", "solution"],
                "follow_up": ["organic_solutions", "traditional_methods"]
            },
            "organic_inquiry": {
                "patterns": [
                    "organic", "natural", "chemical free", "eco friendly",
                    "sustainable", "bio pesticide", "organic fertilizer"
                ],
                "responses": [
                    "Excellent choice going organic! 🌱\n\nBenefits you'll see:\n✓ Cost savings: ₹1,500-2,000/acre\n✓ Better soil health\n✓ Safer for you and environment\n✓ Premium market price\n\nOur organic solutions include:\n- Natural pest control\n- Organic fertilizers\n- Composting methods\n- Traditional practices\n\nAll with local ingredients and detailed guides!",
                    "Going organic is great! Here's what we offer:\n\n🌿 **Pest Control**\n- Neem oil, Garlic spray\n- Panchagavya\n- Bio-pesticides\n\n🌾 **Fertilizers**\n- Vermicompost\n- FYM, Green manure\n- Jeevamrutham\n\n💰 **Cost Comparison**\n- Chemical: ₹3,000-5,000/acre\n- Organic: ₹800-1,500/acre\n- Savings: 50-70%\n\nWant to see detailed preparation methods?"
                ],
                "context": ["organic", "natural"],
                "follow_up": ["organic_solutions", "cost_inquiry"]
            },
            "weather_inquiry": {
                "patterns": [
                    "weather", "rain", "temperature", "forecast", "climate",
                    "rainfall", "will it rain", "hot", "cold"
                ],
                "responses": [
                    "Let me help you with weather information!\n\n🌤️ **Current Weather**\n- Real-time conditions\n- Temperature, humidity\n- Wind speed\n\n📅 **7-Day Forecast**\n- Daily predictions\n- Rainfall alerts\n- Best days for spraying\n\n⚠️ **Weather Alerts**\n- Extreme temperature warnings\n- Heavy rain alerts\n- Storm notifications\n\nCheck the Weather section for detailed information. Would you like me to fetch today's weather for your location?",
                    "Weather is crucial for farming! Our weather system provides:\n\n✓ Current conditions\n✓ 7-day forecast\n✓ Rainfall predictions\n✓ Temperature alerts\n✓ Farming advisories based on weather\n\nExample: If rain is expected, we'll suggest:\n- Postpone spraying\n- Ensure drainage\n- Protect young plants\n\nWhat's your location? I can get specific weather info!"
                ],
                "context": ["weather", "forecast"],
                "follow_up": ["seasonal_inquiry"]
            },
            "seasonal_inquiry": {
                "patterns": [
                    "season", "when to plant", "sowing time", "harvest time",
                    "crop calendar", "kharif", "rabi", "summer crop", "best time"
                ],
                "responses": [
                    "Seasonal planning is key! 📅\n\n**Current Season Info:**\n\n🌾 **Kharif** (June-October)\n- Rice, Cotton, Soybean, Maize\n- Monsoon crops\n\n❄️ **Rabi** (November-February)\n- Wheat, Chickpea, Mustard\n- Winter crops\n\n☀️ **Summer** (March-May)\n- Watermelon, Vegetables\n- Irrigation critical\n\nVisit 'Seasonal Calendar' for:\n- Month-wise activities\n- Crop recommendations\n- Disease alerts\n- Best practices\n\nWhat crops are you interested in?",
                    "Let me help you with seasonal planning!\n\nFor optimal results, check:\n\n1️⃣ **Seasonal Calendar**\n   - Best planting dates\n   - Harvest timelines\n   - Monthly activities\n\n2️⃣ **Crop Recommendations**\n   - Suitable crops for your region\n   - Expected yields\n   - Disease resistance\n\n3️⃣ **Weather Patterns**\n   - Historical data\n   - Best sowing windows\n\nTell me your district and I'll give specific recommendations!"
                ],
                "context": ["seasonal", "calendar"],
                "follow_up": ["crop_recommendation"]
            },
            "specialist_inquiry": {
                "patterns": [
                    "specialist", "expert", "consultation", "advice",
                    "talk to doctor", "need help", "agricultural expert"
                ],
                "responses": [
                    "Connect with agricultural specialists! 👨‍⚕️\n\n**Available Options:**\n\n💬 **Chat Consultation** (Free)\n- Instant messaging\n- Start anytime\n- Share photos\n\n📹 **Video Consultation** (By appointment)\n- Face-to-face guidance\n- Real-time diagnosis\n- Book in advance\n\n**Our Specialists:**\n- 50+ verified experts\n- 15+ years experience\n- Multiple languages\n- Crop disease specialists\n\nGo to 'Consultations' section to connect. Want me to show available specialists?",
                    "Expert help is just a click away!\n\n🔍 **Find Specialists by:**\n- Specialization (diseases, pests, organic)\n- Language (Telugu, English, Hindi)\n- Experience level\n- Rating & reviews\n\n📊 **Success Rate:**\n- 95% satisfaction\n- Average response: 5 minutes\n- 1000+ consultations completed\n\nReady to connect? I can show you online specialists now!"
                ],
                "context": ["specialist", "expert"],
                "follow_up": ["consultation_booking"]
            },
            "community_inquiry": {
                "patterns": [
                    "community", "forum", "farmers", "discuss", "ask question",
                    "share experience", "other farmers", "post"
                ],
                "responses": [
                    "Join our vibrant farming community! 👥\n\n**Community Features:**\n\n💬 **Discussion Forum**\n- Ask questions\n- Get answers from 5000+ farmers\n- Share experiences\n\n📸 **Share Crop Photos**\n- Before/after treatments\n- Success stories\n- Learn from others\n\n🏆 **Achievements**\n- Most helpful farmer\n- Top contributor\n- Expert badge\n\n🔥 **Active Topics:**\n- Organic pest control\n- Seasonal tips\n- Disease outbreaks\n- Market prices\n\nVisit Community section to join discussions!",
                    "Our community has 5000+ farmers helping each other! 🌾\n\n**Recent Success Stories:**\n\n✅ Ramesh from Guntur\n   - Solved bollworm issue organically\n   - Shared neem spray recipe\n   - Helped 50+ farmers\n\n✅ Lakshmi from Vizag\n   - Increased yield by 30%\n   - Posted step-by-step guide\n   - Community star ⭐\n\n**You can:**\n- Post questions\n- Share photos\n- Help others\n- Earn badges\n\nReady to join? Click Community!"
                ],
                "context": ["community", "forum"],
                "follow_up": ["post_creation"]
            },
            "video_inquiry": {
                "patterns": [
                    "video", "tutorial", "learn", "watch", "how to video",
                    "demonstration", "show me", "youtube"
                ],
                "responses": [
                    "Learn through our video library! 📺\n\n**Video Categories:**\n\n🐛 **Pest Control** (25 videos)\n- Neem oil preparation\n- Garlic chili spray\n- Organic pesticides\n\n🌱 **Composting** (18 videos)\n- Vermicompost making\n- Pit composting\n- Drum composting\n\n💧 **Irrigation** (15 videos)\n- Drip irrigation setup\n- Water conservation\n- Mulching techniques\n\n🌾 **Crop Management** (30 videos)\n- Planting methods\n- Pruning techniques\n- Harvest timing\n\n**Languages:** Telugu, English, Hindi\n**Duration:** 5-20 minutes\n\nWhich topic interests you?",
                    "50+ video tutorials waiting for you! 🎥\n\n**Most Popular:**\n\n1️⃣ Vermicompost Preparation (15 min)\n   - 10,000+ views\n   - Step-by-step guide\n   - Telugu language\n\n2️⃣ Neem Oil Spray (10 min)\n   - 8,500+ views\n   - Pest control method\n   - Both languages\n\n3️⃣ Drip Irrigation Setup (18 min)\n   - 7,200+ views\n   - Water saving technique\n\n**Track Progress:**\n- Watch history\n- Completion badges\n- Save favorites\n\nVisit Video Tutorials section!"
                ],
                "context": ["video", "learning"],
                "follow_up": ["learning_inquiry"]
            },
            "traditional_inquiry": {
                "patterns": [
                    "traditional", "tribal", "old method", "ancestors",
                    "natural way", "indigenous", "folk knowledge"
                ],
                "responses": [
                    "Discover ancient farming wisdom! 📜\n\n**Traditional Practices from:**\n\n🌿 **Koya Tribe**\n- Moon phase planting\n- Natural pest deterrents\n- Soil enrichment methods\n\n🌾 **Savara Tribe**\n- Companion planting\n- Seed treatment\n- Water conservation\n\n🍃 **Chenchu Community**\n- Forest-based solutions\n- Herbal preparations\n- Sustainable practices\n\n**All Methods:**\n✓ Verified by tribal elders\n✓ Scientifically explained\n✓ Proven over generations\n✓ Eco-friendly & sustainable\n\nExplore Traditional Knowledge section!",
                    "Traditional wisdom meets modern farming! 🌳\n\n**Popular Traditional Methods:**\n\n1️⃣ **Moon Phase Planting**\n   - Plant according to lunar cycle\n   - 30% higher germination\n   - Used for centuries\n\n2️⃣ **Ash & Charcoal Pest Control**\n   - Natural insect deterrent\n   - Improves soil pH\n   - Zero cost\n\n3️⃣ **Termite Mound Soil**\n   - Rich in nutrients\n   - Natural fertilizer\n   - Free from nature\n\n**Scientific Basis Included!**\nEach method explained with modern science.\n\nWant to learn specific traditional practices?"
                ],
                "context": ["traditional", "tribal"],
                "follow_up": ["organic_inquiry"]
            },
            "cost_inquiry": {
                "patterns": [
                    "cost", "price", "cheap", "affordable", "budget",
                    "save money", "expensive", "how much"
                ],
                "responses": [
                    "Let's talk about organic farming costs! 💰\n\n**Cost Comparison:**\n\n🔴 **Chemical Farming** (per acre)\n- Pesticides: ₹2,000-3,000\n- Fertilizers: ₹3,000-4,000\n- Total: ₹5,000-7,000\n\n🟢 **Organic Farming** (per acre)\n- Neem spray: ₹150-250\n- Vermicompost: ₹300-500\n- Traditional methods: ₹100-200\n- Total: ₹800-1,500\n\n💵 **Savings: 50-70%!**\n\n**Additional Benefits:**\n✓ Better soil health\n✓ Premium market prices (+20-30%)\n✓ Long-term sustainability\n✓ Lower health risks\n\nCheck your Impact Dashboard to see your personal savings!",
                    "Organic farming is surprisingly affordable! 🌾\n\n**Sample Budget (1 acre):**\n\n**Neem Oil Spray:**\n- Neem oil: ₹200\n- Soap: ₹50\n- Labor: Free (DIY)\n- Total: ₹250\n- Lasts: 10-15 days\n\n**Vermicompost:**\n- One-time setup: ₹500\n- Kitchen waste: Free\n- Maintenance: Minimal\n- Production: 50kg/month\n- Value: ₹1,500/month\n\n**ROI: 300% in 6 months!**\n\nWant detailed cost breakdown for specific solutions?"
                ],
                "context": ["cost", "price"],
                "follow_up": ["organic_inquiry", "roi_inquiry"]
            },
            "upload_inquiry": {
                "patterns": [
                    "upload", "photo", "picture", "image", "scan",
                    "take picture", "camera", "analyze photo"
                ],
                "responses": [
                    "Let me guide you to upload crop photos! 📸\n\n**Step-by-Step:**\n\n1️⃣ Go to 'Crop Analysis' section\n2️⃣ Click 'Upload Photo' button\n3️⃣ Take or select photo:\n   - Well-lit conditions\n   - Focus on affected area\n   - Clear, not blurry\n4️⃣ AI analyzes in 5 seconds\n5️⃣ Get diagnosis + treatment\n\n**Tips for Best Results:**\n✓ Natural daylight\n✓ Multiple angles\n✓ Close-up of symptoms\n✓ Include healthy parts too\n\n**What Happens Next:**\n- Disease identified\n- Confidence score shown\n- Organic treatment suggested\n- Track in history\n\nReady to upload? Go to home screen!",
                    "Upload your crop photos for instant AI diagnosis! 🔬\n\n**Our AI Can Detect:**\n- 50+ common diseases\n- Pest infestations\n- Nutrient deficiencies\n- Stress symptoms\n\n**Accuracy: 95%+**\n\n**Photo Guidelines:**\n\n✅ **Good Photos:**\n- Clear focus\n- Good lighting\n- Affected area visible\n- 5-10 cm distance\n\n❌ **Avoid:**\n- Blurry images\n- Too dark/bright\n- Too far away\n- Multiple plants\n\n**After Upload:**\n1. Instant analysis\n2. Disease name\n3. Severity level\n4. Treatment options\n5. Success probability\n\nStart from Crop Analysis tab!"
                ],
                "context": ["upload", "photo"],
                "follow_up": ["disease_inquiry", "treatment_inquiry"]
            },
            "products_inquiry": {
                "patterns": [
                    "buy", "product", "shop", "purchase", "seed", "tool",
                    "equipment", "marketplace", "supplier"
                ],
                "responses": [
                    "Browse our organic marketplace! 🛒\n\n**Available Products:**\n\n🌱 **Seeds**\n- Organic varieties\n- Hybrid options\n- Traditional seeds\n- ₹150-500/packet\n\n🌿 **Fertilizers**\n- Vermicompost\n- Neem cake\n- Bio-fertilizers\n- ₹200-500/bag\n\n🔧 **Tools & Equipment**\n- Hand tools\n- Sprayers\n- Drip irrigation kits\n- ₹300-8,000\n\n🍃 **Organic Inputs**\n- Neem oil\n- Bio-pesticides\n- Growth promoters\n\n**Benefits:**\n✓ Direct from suppliers\n✓ Verified quality\n✓ Competitive prices\n✓ Local delivery\n\nVisit Marketplace section to browse!",
                    "Shop from verified local suppliers! 🏪\n\n**Featured Products:**\n\n1️⃣ **Organic Tomato Seeds**\n   - Price: ₹150/100g\n   - Location: Visakhapatnam\n   - Rating: 4.8⭐\n   - 85% germination\n\n2️⃣ **Premium Vermicompost**\n   - Price: ₹300/10kg\n   - Location: Guntur\n   - Rating: 4.9⭐\n   - Rich NPK content\n\n3️⃣ **Neem Oil Concentrate**\n   - Price: ₹250/liter\n   - Location: Anantapur\n   - Rating: 4.7⭐\n   - 1500ppm azadirachtin\n\n**Secure Payments:**\n- Stripe integrated\n- Cash on delivery\n- Buyer protection\n\nReady to shop?"
                ],
                "context": ["products", "marketplace"],
                "follow_up": ["purchase_inquiry"]
            },
            "success_inquiry": {
                "patterns": [
                    "success", "result", "work", "effective", "proof",
                    "testimonial", "review", "does it work"
                ],
                "responses": [
                    "Real results from real farmers! 🏆\n\n**Platform Impact:**\n\n📊 **Overall Success Rate: 85%**\n\n✓ 10,000+ farmers helped\n✓ 50,000+ crops analyzed\n✓ 5,000+ successful treatments\n✓ ₹5-10 crores total savings\n\n**Recent Success Stories:**\n\n🌾 **Ramesh - Guntur**\n\"Saved my cotton crop from bollworm using neem spray. Cost: ₹200. Saved: ₹15,000!\"\n- Disease: Bollworm infestation\n- Treatment: Organic neem oil\n- Result: 90% pest reduction\n- Time: 2 weeks\n\n🌱 **Lakshmi - Vizag**\n\"Increased tomato yield by 30% with organic methods. Market price +25%!\"\n- Methods: Vermicompost + companion planting\n- Investment: ₹800\n- Additional income: ₹25,000\n\nCheck Community for 500+ success stories!",
                    "Proven results you can trust! ✅\n\n**Success Metrics:**\n\n🎯 **Disease Detection**\n- Accuracy: 95%+\n- Speed: 5 seconds\n- Success rate: 90%\n\n💊 **Treatment Effectiveness**\n\nNeem Oil Spray:\n- Success: 85%\n- Cost: ₹150/acre\n- Users: 2,500+\n\nVermicompost:\n- Success: 90%\n- Cost: ₹300\n- Users: 3,200+\n\nPanchagavya:\n- Success: 88%\n- Cost: ₹100/acre\n- Users: 1,800+\n\n**Your Progress Tracking:**\n- View Impact Dashboard\n- See personal statistics\n- Track treatments\n- Monitor success rate\n\nWant to see detailed case studies?"
                ],
                "context": ["success", "proof"],
                "follow_up": ["community_inquiry", "testimonials"]
            },
            "help_inquiry": {
                "patterns": [
                    "help", "how", "guide", "tutorial", "show me",
                    "teach", "explain", "confused", "don't understand"
                ],
                "responses": [
                    "I'm here to help! Let me guide you. 🤝\n\n**Quick Start Guide:**\n\n1️⃣ **For Disease Identification:**\n   Go to: Crop Analysis → Upload Photo\n   Get: Instant diagnosis + treatment\n\n2️⃣ **For Organic Solutions:**\n   Go to: Organic Solutions → Browse by crop\n   Get: Step-by-step recipes\n\n3️⃣ **For Expert Help:**\n   Go to: Consultations → Find Specialist\n   Get: Chat or video consultation\n\n4️⃣ **For Learning:**\n   Go to: Video Tutorials → Select topic\n   Get: Visual demonstrations\n\n5️⃣ **For Community:**\n   Go to: Community Forum → Ask/Share\n   Get: Farmer experiences\n\n**What would you like help with specifically?**\nJust tell me and I'll guide you step-by-step!",
                    "Let me help you navigate the platform! 🗺️\n\n**Main Features:**\n\n🏠 **Dashboard**\n- Overview of your farm\n- Recent analyses\n- Weather alerts\n- Quick actions\n\n🔬 **Crop Analysis**\n- Upload photos\n- AI diagnosis\n- Treatment suggestions\n- History tracking\n\n🌿 **Organic Solutions**\n- 100+ treatments\n- Detailed recipes\n- Cost information\n- Success rates\n\n👨‍⚕️ **Consultations**\n- Find specialists\n- Chat support\n- Video calls\n- Expert advice\n\n👥 **Community**\n- Ask questions\n- Share experiences\n- Learn from others\n\n📺 **Video Tutorials**\n- Learn visually\n- Step-by-step guides\n\n**Need help with something specific?**\nTell me what you're trying to do!"
                ],
                "context": ["help", "guide"],
                "follow_up": ["navigation_help"]
            },
            "language_inquiry": {
                "patterns": [
                    "language", "telugu", "english", "hindi", "translate",
                    "change language", "regional language"
                ],
                "responses": [
                    "We support multiple languages! 🌐\n\n**Available Languages:**\n\n✅ **Telugu (తెలుగు)**\n- Full platform support\n- All videos available\n- Community posts\n- Chat support\n\n✅ **English**\n- Full platform support\n- All content\n- International standard\n\n🔄 **Coming Soon:**\n- Hindi (हिंदी)\n- Kannada (ಕನ್ನಡ)\n- Tamil (தமிழ்)\n\n**To Change Language:**\n1. Go to Profile/Settings\n2. Select Language Preference\n3. Choose your language\n4. Restart app (if needed)\n\n**Content in Your Language:**\n- Video tutorials\n- Treatment guides\n- Community discussions\n- Expert consultations\n- All features\n\nWhich language would you prefer?",
                    "మీ భాషలో వినండి! 🗣️\n\nLanguage support available:\n\n📱 **Platform:**\n- Telugu (Full)\n- English (Full)\n- Hindi (Partial)\n\n🎥 **Videos:**\n- 80% in Telugu\n- 100% in English\n- Subtitles available\n\n👥 **Community:**\n- Post in any language\n- Auto-translation coming\n\n👨‍⚕️ **Specialists:**\n- Telugu speakers: 40+\n- English speakers: 50+\n- Hindi speakers: 25+\n\n**Change anytime from Settings!**\n\nప్రశ్నలు ఉన్నాయా? (Have questions?)"
                ],
                "context": ["language", "translation"],
                "follow_up": ["settings_help"]
            },
            "contact_inquiry": {
                "patterns": [
                    "contact", "support", "phone", "email", "reach",
                    "call", "customer care", "helpline"
                ],
                "responses": [
                    "Multiple ways to get support! 📞\n\n**Instant Support:**\n💬 **This Chat**\n- Available 24/7\n- Instant responses\n- No waiting\n\n**Expert Support:**\n👨‍⚕️ **Specialists**\n- Book consultation\n- Chat or video call\n- Agricultural experts\n\n**Community Support:**\n👥 **Forum**\n- Ask 5000+ farmers\n- Shared experiences\n- Real solutions\n\n**Technical Support:**\n📧 Email: support@organicadvisory.com\n📱 Phone: +91-XXXXX-XXXXX\n⏰ Hours: 9 AM - 6 PM (Mon-Sat)\n\n**Help Center:**\n❓ FAQs section\n📚 User guides\n🎥 Tutorial videos\n\n**How can I help you right now?**",
                    "We're here to support you! 🤝\n\n**Support Channels:**\n\n1️⃣ **In-App Chat** (You're here!)\n   - Instant help\n   - Any time\n   - Any question\n\n2️⃣ **Specialist Consultation**\n   - 50+ experts available\n   - Chat: Free\n   - Video: By appointment\n   - Response time: 5-10 min\n\n3️⃣ **Community Forum**\n   - 5000+ active farmers\n   - Real experiences\n   - Quick responses\n\n4️⃣ **Help Center**\n   - FAQs\n   - Guides\n   - Troubleshooting\n\n5️⃣ **Emergency Contact**\n   - Agricultural dept. helpline\n   - Pest outbreak alerts\n   - Disease emergency\n\n**What do you need help with?**"
                ],
                "context": ["contact", "support"],
                "follow_up": ["specialist_inquiry", "help_inquiry"]
            },
            "thanks": {
                "patterns": [
                    "thank", "thanks", "appreciate", "grateful",
                    "helpful", "good", "nice"
                ],
                "responses": [
                    "You're very welcome! 😊 I'm glad I could help.\n\nFeel free to ask me anything else about:\n- Crop diseases and treatments\n- Organic farming methods\n- Weather and seasonal advice\n- Expert consultations\n- Any farming questions\n\nHappy farming! 🌾",
                    "My pleasure! 🌱 I'm always here to help.\n\nRemember:\n- Upload photos for disease diagnosis\n- Check organic solutions library\n- Connect with specialists anytime\n- Join community discussions\n\nGood luck with your crops! 👨‍🌾",
                    "Glad I could assist you! 🤗\n\nDon't hesitate to reach out if you need:\n- Disease identification\n- Treatment advice\n- Farming tips\n- Expert consultation\n\nWishing you bountiful harvests! 🌾"
                ],
                "context": ["thanks", "gratitude"],
                "follow_up": []
            },
            "goodbye": {
                "patterns": [
                    "bye", "goodbye", "see you", "later", "gtg",
                    "have to go", "thanks bye"
                ],
                "responses": [
                    "Goodbye! Feel free to chat anytime you need help. Happy farming! 🌾👋",
                    "Take care! I'm here 24/7 whenever you need assistance. Good luck! 🌱",
                    "See you later! May your crops grow healthy and strong! 🌿✨"
                ],
                "context": ["goodbye"],
                "follow_up": []
            },
            "my_crops": {
                "patterns": [
                    "my crops", "show my crops", "crops monitored", "what crops",
                    "crop history", "my photos", "uploaded photos", "my analyses"
                ],
                "responses": [
                    "Let me fetch your crop monitoring history! 📊",
                    "Here's your crop analysis data! 🌾"
                ],
                "context": ["crops", "history"],
                "follow_up": []
            },
            "my_badges": {
                "patterns": [
                    "my badges", "achievements", "rewards", "earned badges",
                    "my progress", "my stats", "my achievements", "what badges"
                ],
                "responses": [
                    "Let me show your achievements! 🏆",
                    "Here are your farming badges and progress! ⭐"
                ],
                "context": ["badges", "achievements"],
                "follow_up": []
            },
            "my_purchases": {
                "patterns": [
                    "my orders", "purchases", "bought products", "order history",
                    "what did i buy", "my products", "purchased items", "order status"
                ],
                "responses": [
                    "Let me fetch your purchase history! 🛒",
                    "Here are your orders! 📦"
                ],
                "context": ["orders", "purchases"],
                "follow_up": []
            },
            "my_weather": {
                "patterns": [
                    "my weather", "weather for my location", "local weather",
                    "weather in my area", "my forecast", "weather today"
                ],
                "responses": [
                    "Let me get the weather for your location! ☀️",
                    "Checking weather conditions in your area! 🌤️"
                ],
                "context": ["weather", "local"],
                "follow_up": []
            },
            "my_treatments": {
                "patterns": [
                    "my treatments", "applied treatments", "treatment history",
                    "solutions applied", "what treatments", "my solutions"
                ],
                "responses": [
                    "Let me show your treatment applications! 💊",
                    "Here's your organic treatment history! 🌿"
                ],
                "context": ["treatments", "history"],
                "follow_up": []
            },
            "my_community": {
                "patterns": [
                    "my posts", "my questions", "community activity",
                    "my discussions", "posted questions", "my comments"
                ],
                "responses": [
                    "Let me fetch your community activity! 👥",
                    "Here are your community posts and discussions! 💬"
                ],
                "context": ["community", "posts"],
                "follow_up": []
            },
            "my_consultations": {
                "patterns": [
                    "my consultations", "consultation history", "specialist sessions",
                    "expert meetings", "past consultations", "consultation records"
                ],
                "responses": [
                    "Let me get your consultation history! 👨‍⚕️",
                    "Here are your specialist consultations! 🩺"
                ],
                "context": ["consultations", "history"],
                "follow_up": []
            },
            
            }
        
        
        
        # Context tracking for conversational flow
        self.context_memory = {}
        
        # Initialize vectorizer for intent matching
        self.vectorizer = None
        self.intent_vectors = None
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer with all patterns"""
        all_patterns = []
        for intent_data in self.intents.values():
            all_patterns.extend(intent_data["patterns"])
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=500,
            stop_words='english'
        )
        self.intent_vectors = self.vectorizer.fit_transform(all_patterns)
    
    def get_intent(self, user_message: str) -> Tuple[str, float]:
        """
        Match user message to best intent using TF-IDF + cosine similarity
        Returns: (intent_name, confidence_score)
        """
        message_lower = user_message.lower().strip()
        
        # Transform user message
        message_vector = self.vectorizer.transform([message_lower])
        
        # Calculate similarities with all patterns
        similarities = cosine_similarity(message_vector, self.intent_vectors)[0]
        
        # Find best matching intent
        max_similarity = 0
        best_intent = "help_inquiry"  # default fallback
        
        current_idx = 0
        for intent_name, intent_data in self.intents.items():
            num_patterns = len(intent_data["patterns"])
            intent_similarities = similarities[current_idx:current_idx + num_patterns]
            
            max_intent_sim = max(intent_similarities) if len(intent_similarities) > 0 else 0
            
            if max_intent_sim > max_similarity:
                max_similarity = max_intent_sim
                best_intent = intent_name
            
            current_idx += num_patterns
        
        return best_intent, float(max_similarity)
    
    def get_response(self, intent: str, user_id: str = None) -> str:
        """Get response for matched intent"""
        if intent not in self.intents:
            intent = "help_inquiry"
        
        responses = self.intents[intent]["responses"]
        
        # Return random response from list
        import random
        return random.choice(responses)
    
    def update_context(self, user_id: str, context: List[str]):
        """Update conversation context for user"""
        self.context_memory[user_id] = {
            "context": context,
            "timestamp": datetime.utcnow()
        }
    
    def get_context(self, user_id: str) -> List[str]:
        """Get current conversation context"""
        if user_id in self.context_memory:
            user_context = self.context_memory[user_id]
            # Context expires after 10 minutes
            if (datetime.utcnow() - user_context["timestamp"]).seconds < 600:
                return user_context["context"]
        return []


# ============= CHATBOT CLASS =============

class OrganicFarmingChatbot:
    def __init__(self, db):
        self.db = db
        self.knowledge_base = ChatbotKnowledgeBase()
        self.conversation_history = {}
    
    async def get_user_context(self, user_id: str) -> Dict:
        """Get user-specific context from database"""
        try:
            user = await self.db.users.find_one({"_id": ObjectId(user_id)})
            if not user:
                return {}
            
            # Get user stats
            progress = await self.db.user_progress.find_one({"user_id": user_id})
            
            # Recent photos
            recent_photos = await self.db.crop_photos.find(
                {"user_id": user_id}
            ).sort("uploaded_at", -1).limit(3).to_list(3)
            
            # Recent diseases detected
            recent_diseases = [p.get("disease") for p in recent_photos if p.get("disease")]
            
            return {
                "name": user.get("name", "Farmer"),
                "district": user.get("district", ""),
                "crops_monitored": progress.get("crops_monitored", 0) if progress else 0,
                "treatments_applied": progress.get("treatments_applied", 0) if progress else 0,
                "recent_diseases": list(set(recent_diseases)),
                "language": user.get("language_preference", "telugu")
            }
        except Exception as e:
            print(f"Error getting user context: {e}")
            return {}
    
    async def personalize_response(self, response: str, user_context: Dict) -> str:
        """Add personalization to response based on user context"""
        # Add user name if greeting
        if "Hello!" in response or "Hi there!" in response:
            name = user_context.get("name", "")
            if name:
                response = response.replace("Hello!", f"Hello {name}!")
                response = response.replace("Hi there!", f"Hi {name}!")
        
        # Add location-specific info
        district = user_context.get("district", "")
        if district and "your location" in response.lower():
            response = response.replace("your location", district)
        
        # Add personalized stats
        crops = user_context.get("crops_monitored", 0)
        treatments = user_context.get("treatments_applied", 0)
        
        if crops > 0 and "impact dashboard" in response.lower():
            response += f"\n\n📊 Your Progress:\n- {crops} crops monitored\n- {treatments} treatments applied"
        
        return response
    
    async def get_dynamic_data(self, intent: str, user_id: str) -> Optional[str]:
        """Fetch real-time data from database based on intent"""
        try:
            if intent == "weather_inquiry" or intent == "my_weather":
                # Get weather alerts for user's location
                user = await self.db.users.find_one({"_id": ObjectId(user_id)})
                if user and user.get("district"):
                    alerts = await self.db.weather_alerts.find({
                        "location": {"$regex": user["district"], "$options": "i"}
                    }).sort("timestamp", -1).limit(3).to_list(3)
                    
                    if alerts:
                        alert_text = "\n\n🔔 **Current Alerts for Your Area:**\n"
                        for alert in alerts:
                            alert_text += f"- {alert['alert_type'].title()}: {alert['message']}\n"
                        return alert_text
            
            elif intent == "my_crops":
                # Get user's crop analyses
                photos = await self.db.crop_photos.find({
                    "user_id": user_id
                }).sort("uploaded_at", -1).limit(10).to_list(10)
                
                if photos:
                    crop_text = "\n\n📊 **Your Recent Crop Analyses:**\n\n"
                    for i, photo in enumerate(photos, 1):
                        disease = photo.get("disease", "Unknown")
                        confidence = int(photo.get("confidence_score", 0) * 100)
                        date = photo["uploaded_at"].strftime("%b %d")
                        status = photo.get("status", "active")
                        
                        status_emoji = "✅" if status == "resolved" else "🔄" if status == "treated" else "⚠️"
                        
                        crop_text += f"{i}. {status_emoji} **{disease}** ({confidence}% confident)\n"
                        crop_text += f"   Date: {date} | Status: {status.title()}\n\n"
                    
                    crop_text += f"📈 Total analyses: {len(photos)}\n"
                    crop_text += "💡 Tip: Click on 'Crop Analysis' to see detailed history"
                    return crop_text
                else:
                    return "\n\n📸 You haven't uploaded any crop photos yet. Upload one to get started!"
            
            elif intent == "my_badges":
                # Get user badges and progress
                user = await self.db.users.find_one({"_id": ObjectId(user_id)})
                progress = await self.db.user_progress.find_one({"user_id": user_id})
                
                if user or progress:
                    badge_text = "\n\n🏆 **Your Achievements:**\n\n"
                    
                    # Show badges
                    badges = user.get("badges", []) if user else []
                    if badges:
                        badge_text += "**Earned Badges:**\n"
                        for badge in badges:
                            badge_text += f"⭐ {badge}\n"
                    else:
                        badge_text += "No badges earned yet. Keep farming! 🌱\n"
                    
                    badge_text += "\n**Your Stats:**\n"
                    if progress:
                        badge_text += f"🌾 Crops Monitored: {progress.get('crops_monitored', 0)}\n"
                        badge_text += f"💊 Treatments Applied: {progress.get('treatments_applied', 0)}\n"
                        badge_text += f"✅ Success Rate: {progress.get('success_rate', 0):.1f}%\n"
                        badge_text += f"📚 Learning Sessions: {progress.get('learning_sessions_completed', 0)}\n"
                    
                    # Streak
                    if user:
                        streak = user.get("streak_count", 0)
                        badge_text += f"🔥 Active Streak: {streak} days\n"
                    
                    badge_text += "\n💡 Keep using the platform to earn more badges!"
                    return badge_text
                else:
                    return "\n\n📊 Start using the platform to track your progress!"
            
            elif intent == "my_purchases":
                # Get user's order history
                orders = await self.db.product_orders.find({
                    "buyer_id": user_id
                }).sort("created_at", -1).limit(10).to_list(10)
                
                if orders:
                    order_text = "\n\n🛒 **Your Recent Orders:**\n\n"
                    for i, order in enumerate(orders, 1):
                        product = order.get("product_title", "Unknown Product")
                        price = order.get("total_price", 0)
                        status = order.get("order_status", "pending")
                        date = order["created_at"].strftime("%b %d")
                        
                        status_emoji = "✅" if status == "delivered" else "📦" if status == "shipped" else "⏳"
                        
                        order_text += f"{i}. {status_emoji} **{product}**\n"
                        order_text += f"   ₹{price} | {status.title()} | {date}\n\n"
                    
                    order_text += f"📊 Total orders: {len(orders)}\n"
                    order_text += "🛍️ Visit Marketplace to shop more!"
                    return order_text
                else:
                    return "\n\n🛒 You haven't purchased anything yet. Check out our Marketplace!"
            
            elif intent == "my_treatments":
                # Get applied treatments/solutions
                solutions = await self.db.solution_applications.find({
                    "user_id": user_id
                }).sort("applied_at", -1).limit(10).to_list(10)
                
                if solutions:
                    treatment_text = "\n\n💊 **Your Treatment Applications:**\n\n"
                    for i, sol in enumerate(solutions, 1):
                        sol_data = await self.db.organic_solutions.find_one({
                            "_id": ObjectId(sol["solution_id"])
                        })
                        
                        title = sol_data["title"] if sol_data else "Unknown Treatment"
                        status = sol.get("status", "applied")
                        outcome = sol.get("outcome", "pending")
                        date = sol["applied_at"].strftime("%b %d")
                        
                        status_emoji = "✅" if outcome == "success" else "⏳" if status == "in_progress" else "📋"
                        
                        treatment_text += f"{i}. {status_emoji} **{title}**\n"
                        treatment_text += f"   Status: {status.title()} | {date}\n\n"
                    
                    treatment_text += f"📊 Total treatments: {len(solutions)}\n"
                    treatment_text += "🌿 Check Organic Solutions for more!"
                    return treatment_text
                else:
                    return "\n\n💊 No treatments applied yet. Browse Organic Solutions to start!"
            
            elif intent == "my_community":
                # Get user's community posts
                posts = await self.db.community_posts.find({
                    "author_id": user_id
                }).sort("created_at", -1).limit(10).to_list(10)
                
                if posts:
                    community_text = "\n\n👥 **Your Community Activity:**\n\n"
                    for i, post in enumerate(posts, 1):
                        title = post.get("title", "Untitled")
                        comments_count = len(post.get("comments", []))
                        likes = post.get("likes", 0)
                        solved = post.get("is_solved", False)
                        date = post["created_at"].strftime("%b %d")
                        
                        status_emoji = "✅" if solved else "💬"
                        
                        community_text += f"{i}. {status_emoji} **{title[:40]}...**\n"
                        community_text += f"   💬 {comments_count} comments | 👍 {likes} likes | {date}\n\n"
                    
                    community_text += f"📊 Total posts: {len(posts)}\n"
                    community_text += "👥 Visit Community to see more!"
                    return community_text
                else:
                    return "\n\n👥 You haven't posted in the community yet. Share your experience!"
            
            elif intent == "my_consultations":
                # Get consultation history
                consultations = await self.db.consultation_sessions.find({
                    "farmer_id": user_id
                }).sort("created_at", -1).limit(10).to_list(10)
                
                if consultations:
                    consult_text = "\n\n👨‍⚕️ **Your Consultations:**\n\n"
                    for i, consult in enumerate(consultations, 1):
                        specialist = consult.get("specialist_name", "Unknown Specialist")
                        status = consult.get("status", "pending")
                        session_type = consult.get("session_type", "chat")
                        date = consult["created_at"].strftime("%b %d")
                        
                        status_emoji = "✅" if status == "completed" else "🔄" if status == "active" else "⏳"
                        type_emoji = "📹" if session_type == "video" else "💬"
                        
                        consult_text += f"{i}. {status_emoji} {type_emoji} **{specialist}**\n"
                        consult_text += f"   Status: {status.title()} | {date}\n\n"
                    
                    consult_text += f"📊 Total consultations: {len(consultations)}\n"
                    consult_text += "👨‍⚕️ Book more consultations anytime!"
                    return consult_text
                else:
                    return "\n\n👨‍⚕️ You haven't had any consultations yet. Connect with specialists!"
            
            elif intent == "specialist_inquiry":
                # Get online specialists count
                online_count = await self.db.specialist_profiles.count_documents({
                    "is_online": True
                })
                if online_count > 0:
                    return f"\n\n✅ **{online_count} specialists are online right now!**"
            
            elif intent == "disease_inquiry":
                # Get recent disease trends
                pipeline = [
                    {"$group": {"_id": "$disease", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 3}
                ]
                trends = await self.db.crop_photos.aggregate(pipeline).to_list(3)
                
                if trends:
                    trend_text = "\n\n📊 **Trending Issues:**\n"
                    for trend in trends:
                        if trend["_id"]:
                            trend_text += f"- {trend['_id']} ({trend['count']} cases)\n"
                    return trend_text
            
            elif intent == "community_inquiry":
                # Get recent community activity
                recent_posts = await self.db.community_posts.count_documents({
                    "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
                })
                if recent_posts > 0:
                    return f"\n\n🔥 **{recent_posts} new posts in last 24 hours!**"
            
        except Exception as e:
            print(f"Error fetching dynamic data: {e}")
        
        return None
    
    async def process_message(self, user_id: str, message: str) -> Dict:
        """Main method to process user message and generate response"""
        try:
            # Get intent and confidence
            intent, confidence = self.knowledge_base.get_intent(message)
            
            # Get base response
            response = self.knowledge_base.get_response(intent, user_id)
            
            # Get user context for personalization
            user_context = await self.get_user_context(user_id)
            
            # Personalize response
            response = await self.personalize_response(response, user_context)
            
            # Add dynamic data if available
            dynamic_data = await self.get_dynamic_data(intent, user_id)
            if dynamic_data:
                response += dynamic_data
            
            # Update context
            self.knowledge_base.update_context(
                user_id, 
                self.knowledge_base.intents[intent]["context"]
            )
            
            # Store conversation
            await self._store_conversation(user_id, message, response, intent, confidence)
            
            # Get suggested actions
            suggested_actions = self._get_suggested_actions(intent)
            
            return {
                "response": response,
                "intent": intent,
                "confidence": float(confidence),
                "suggested_actions": suggested_actions,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            print(f"Error processing message: {e}")
            return {
                "response": "I'm having trouble understanding. Could you please rephrase your question? I can help you with crop diseases, treatments, weather, and farming advice.",
                "intent": "error",
                "confidence": 0.0,
                "suggested_actions": [
                    {"text": "Talk to specialist", "action": "specialist"},
                    {"text": "Browse solutions", "action": "solutions"},
                    {"text": "Upload photo", "action": "upload"}
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _store_conversation(self, user_id: str, message: str, response: str, intent: str, confidence: float):
        """Store conversation in database for analytics"""
        try:
            await self.db.chatbot_conversations.insert_one({
                "user_id": user_id,
                "user_message": message,
                "bot_response": response,
                "intent": intent,
                "confidence": confidence,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            print(f"Error storing conversation: {e}")
    
    def _get_suggested_actions(self, intent: str) -> List[Dict]:
        """Get suggested quick actions based on intent"""
        actions_map = {
            "disease_inquiry": [
                {"text": "Upload crop photo", "action": "upload", "icon": "📸"},
                {"text": "Browse diseases", "action": "diseases", "icon": "🔬"},
                {"text": "Talk to specialist", "action": "specialist", "icon": "👨‍⚕️"}
            ],
            "treatment_inquiry": [
                {"text": "Organic solutions", "action": "solutions", "icon": "🌱"},
                {"text": "Traditional methods", "action": "traditional", "icon": "📜"},
                {"text": "Video tutorials", "action": "videos", "icon": "📺"}
            ],
            "weather_inquiry": [
                {"text": "View forecast", "action": "weather", "icon": "🌤️"},
                {"text": "Weather alerts", "action": "alerts", "icon": "⚠️"},
                {"text": "Seasonal calendar", "action": "calendar", "icon": "📅"}
            ],
            "my_weather": [
                {"text": "7-day forecast", "action": "weather", "icon": "🌤️"},
                {"text": "Farming advisory", "action": "calendar", "icon": "📅"}
            ],
            "specialist_inquiry": [
                {"text": "Find specialists", "action": "specialists", "icon": "👨‍⚕️"},
                {"text": "Start chat", "action": "chat", "icon": "💬"},
                {"text": "Book video call", "action": "video", "icon": "📹"}
            ],
            "community_inquiry": [
                {"text": "Browse forum", "action": "community", "icon": "👥"},
                {"text": "Ask question", "action": "post", "icon": "❓"},
                {"text": "Success stories", "action": "stories", "icon": "🏆"}
            ],
            "my_crops": [
                {"text": "Upload new photo", "action": "upload", "icon": "📸"},
                {"text": "View all analyses", "action": "analyze", "icon": "🔬"},
                {"text": "Get treatment", "action": "solutions", "icon": "💊"}
            ],
            "my_badges": [
                {"text": "View impact", "action": "impact", "icon": "📊"},
                {"text": "Check leaderboard", "action": "leaderboard", "icon": "🏆"}
            ],
            "my_purchases": [
                {"text": "Shop more", "action": "marketplace", "icon": "🛒"},
                {"text": "Track orders", "action": "orders", "icon": "📦"}
            ],
            "my_treatments": [
                {"text": "Browse solutions", "action": "solutions", "icon": "🌿"},
                {"text": "Track progress", "action": "treatments", "icon": "📊"}
            ],
            "my_community": [
                {"text": "Create new post", "action": "post", "icon": "✍️"},
                {"text": "Browse community", "action": "community", "icon": "👥"}
            ],
            "my_consultations": [
                {"text": "Book consultation", "action": "consultation", "icon": "📅"},
                {"text": "Find specialist", "action": "specialists", "icon": "👨‍⚕️"}
            ]
        }
        
        return actions_map.get(intent, [
            {"text": "Crop analysis", "action": "upload", "icon": "📸"},
            {"text": "Organic solutions", "action": "solutions", "icon": "🌱"},
            {"text": "Talk to expert", "action": "specialist", "icon": "👨‍⚕️"}
        ])


# ============= API ENDPOINTS =============

# Initialize chatbot
chatbot = None

@app.on_event("startup")
async def initialize_chatbot():
    """Initialize chatbot on app startup"""
    global chatbot
    chatbot = OrganicFarmingChatbot(db)
    print("✅ Chatbot initialized successfully")


@app.post("/api/chatbot/message")
async def send_chatbot_message(
    message_data: Dict,
    current_user: dict = Depends(get_current_user)
):
    """
    Send message to chatbot and get response
    Request: {"message": "user message"}
    """
    user_message = message_data.get("message", "").strip()
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(user_message) > 500:
        raise HTTPException(status_code=400, detail="Message too long (max 500 characters)")
    
    user_id = str(current_user["_id"])
    
    # Process message through chatbot
    response = await chatbot.process_message(user_id, user_message)
    
    return response


@app.get("/api/chatbot/history")
async def get_chatbot_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get user's chatbot conversation history"""
    user_id = str(current_user["_id"])
    
    conversations = await db.chatbot_conversations.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    
    return [
        {
            "id": str(conv["_id"]),
            "user_message": conv["user_message"],
            "bot_response": conv["bot_response"],
            "intent": conv.get("intent"),
            "confidence": conv.get("confidence", 0.0),
            "timestamp": conv["timestamp"].isoformat()
        } for conv in conversations
    ]


@app.delete("/api/chatbot/history")
async def clear_chatbot_history(
    current_user: dict = Depends(get_current_user)
):
    """Clear user's chatbot conversation history"""
    user_id = str(current_user["_id"])
    
    result = await db.chatbot_conversations.delete_many({"user_id": user_id})
    
    return {
        "message": f"Deleted {result.deleted_count} conversations",
        "deleted_count": result.deleted_count
    }


@app.get("/api/chatbot/analytics")
async def get_chatbot_analytics(
    current_user: dict = Depends(require_role("specialist"))
):
    """Get chatbot analytics (admin/specialist only)"""
    
    # Total conversations
    total_convs = await db.chatbot_conversations.count_documents({})
    
    # Intent distribution
    intent_pipeline = [
        {"$group": {"_id": "$intent", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    intent_stats = await db.chatbot_conversations.aggregate(intent_pipeline).to_list(10)
    
    # Average confidence
    confidence_pipeline = [
        {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence"}}}
    ]
    confidence_result = await db.chatbot_conversations.aggregate(confidence_pipeline).to_list(1)
    avg_confidence = confidence_result[0]["avg_confidence"] if confidence_result else 0
    
    # Active users (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    active_users = await db.chatbot_conversations.distinct("user_id", {
        "timestamp": {"$gte": week_ago}
    })
    
    return {
        "total_conversations": total_convs,
        "active_users_7d": len(active_users),
        "average_confidence": round(avg_confidence, 2),
        "intent_distribution": [
            {"intent": i["_id"], "count": i["count"]} 
            for i in intent_stats
        ]
    }


@app.post("/api/chatbot/feedback")
async def submit_chatbot_feedback(
    feedback_data: Dict,
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback on chatbot response"""
    conversation_id = feedback_data.get("conversation_id")
    helpful = feedback_data.get("helpful", True)
    comment = feedback_data.get("comment")
    
    if not conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID required")
    
    feedback_doc = {
        "conversation_id": conversation_id,
        "user_id": str(current_user["_id"]),
        "helpful": helpful,
        "comment": comment,
        "timestamp": datetime.utcnow()
    }
    
    await db.chatbot_feedback.insert_one(feedback_doc)
    
    return {"message": "Feedback submitted successfully"}


@app.get("/")
async def root():
    return {"message": "Organic Advisory System API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)