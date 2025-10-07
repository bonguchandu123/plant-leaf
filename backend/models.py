

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict,Any
from datetime import datetime
from bson import ObjectId

# Custom ObjectId type for MongoDB
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


# User Models
class User(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    phone: str
    email: Optional[EmailStr] = None
    password_hash: str
    role: str
    village: Optional[str] = None
    mandal: Optional[str] = None
    district: Optional[str] = None
    profile_image: Optional[str] = None
    language_preference: str = "telugu"
    last_active: Optional[datetime] = None
    badges: List[str] = []
    streak_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserProgress(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    crops_monitored: int = 0
    treatments_applied: int = 0
    success_rate: float = 0.0
    learning_sessions_completed: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class LearningPreferences(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    preferred_content_type: str = "video"
    preferred_language: str = "telugu"
    notifications_enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Crop Models
class Crop(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    local_name: str
    season: str
    soil_type: List[str] = []
    recommended_treatments: List[str] = []
    disease_history: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CropPhoto(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    crop_id: Optional[str] = None
    image_url: str
    location: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    disease: Optional[str] = None
    confidence_score: Optional[float] = None
    suggested_treatment: Optional[str] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Disease & Pest Models
class Disease(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    symptoms: List[str] = []
    affected_crops: List[str] = []
    pest_type: Optional[str] = None
    severity: str
    ai_model_reference: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Pest(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    symptoms: List[str] = []
    affected_crops: List[str] = []
    severity: str
    ai_model_reference: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Treatment Models
class Treatment(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    crop_id: Optional[str] = None
    disease_or_pest_id: Optional[str] = None
    step_by_step_guide: List[str] = []
    ingredients_local_availability: List[dict] = []
    estimated_time: str
    seasonal_relevance: List[str] = []
    media_urls: List[str] = []
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class TreatmentSubmission(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    treatment_id: str
    status: str = "pending"
    outcome: Optional[str] = None
    notes: Optional[str] = None
    photo_before_after: List[str] = []
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Advisory Models
class AdvisoryCalendar(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    crop_id: str
    month: str
    preventive_actions: List[str] = []
    treatment_alerts: List[str] = []
    weather_alerts: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}



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



class WeatherAdvisoryCreateRequest(BaseModel):
    title: str
    advisory_type: str
    priority: str
    description: str
    recommended_actions: List[str]
    target_crops: Optional[List[str]] = None
    target_region: str
    target_districts: Optional[List[str]] = None
    weather_conditions: Optional[dict] = None
    valid_from: str
    valid_until: str


class WeatherAlertCreateRequest(BaseModel):
    location: str
    crop_id: Optional[str] = None
    alert_type: str
    message: str
    recommended_action: str
    severity: Optional[str] = "medium"
    affected_crops: Optional[List[str]] = None
    valid_until: Optional[str] = None


# Community Models
# Add these models to your models.py file

# Enhanced Community Models with Analysis Integration

class CommunityPost(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    content_text: str
    media_urls: List[str] = []
    author_id: str
    author_name: str  # Added for easier display
    location: Optional[str] = None
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    comments: List[str] = []  # Comment IDs
    
    # Engagement metrics
    likes: int = 0
    views: int = 0
    helpful_count: int = 0
    
    # Analysis reference (when shared from crop analysis)
    analysis_reference: Optional[Dict] = None  # {photo_id, image_url, disease, confidence, suggested_treatment}
    
    # Question/Answer features
    is_question: bool = False
    is_solved: bool = False

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CommunityComment(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    post_id: str
    user_id: str
    user_name: str  # Added for easier display
    comment_text: str
    media_urls: List[str] = []
    likes: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class PostLike(BaseModel):
    """Track who liked which posts"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    post_id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CommentLike(BaseModel):
    """Track who liked which comments"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    comment_id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Enhanced CropPhoto model - update existing one
class CropPhoto(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    crop_id: Optional[str] = None
    image_url: str
    location: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    disease: Optional[str] = None
    confidence_score: Optional[float] = None
    suggested_treatment: Optional[str] = None
    
    # Enhanced fields
    severity: str = "medium"  # low, medium, high
    status: str = "active"  # active, treated, resolved
    notes: Optional[str] = None
    updated_at: Optional[datetime] = None
    
    # Community integration
    community_post_id: Optional[str] = None  # Link to community post if shared

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request Models for Community
class CommunityPostCreateRequest(BaseModel):
    title: str
    content_text: str
    tags: List[str] = []
    photo_id: Optional[str] = None  # Reference to crop analysis
    media_urls: List[str] = []
    is_question: bool = False


class CommunityPostUpdateRequest(BaseModel):
    title: Optional[str] = None
    content_text: Optional[str] = None
    tags: Optional[List[str]] = None
    is_solved: Optional[bool] = None


class CommentCreateRequest(BaseModel):
    comment_text: str
    media_urls: List[str] = []


class ShareAnalysisRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = []


# WebSocket Message Models
class WebSocketMessage(BaseModel):
    type: str  # new_post, new_comment, typing, etc.
    data: Dict = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    sender_id: str
    sender_name: Optional[str] = None  # Added for display
    receiver_id: Optional[str] = None
    group_id: Optional[str] = None
    message_text: str
    file_url: Optional[str] = None
    image_url: Optional[str] = None
    message_type: str = "text"  # text, image, file, system
    read: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class FarmerGroup(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    description: Optional[str] = None
    members: List[str] = []  # User IDs
    admins: List[str] = []  # User IDs with admin privileges
    focus_crop: Optional[str] = None
    location: Optional[str] = None
    chat_history: List[str] = []  # Message IDs
    shared_resources: List[str] = []  # Resource IDs
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Notification Models
class Notification(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    notification_type: str  # comment, like, mention, reply, system
    title: str
    message: str
    reference_id: Optional[str] = None  # ID of related post/comment/etc
    reference_type: Optional[str] = None  # post, comment, analysis, etc
    link: Optional[str] = None
    read: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class NotificationCreate(BaseModel):
    user_id: str
    type: str  # info, alert, announcement, system
    title: str
    message: str
    link: Optional[str] = None
    metadata: Optional[dict] = None


# Activity Log
class UserActivity(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    activity_type: str  # post_created, comment_added, analysis_shared, etc
    description: str
    reference_id: Optional[str] = None
    reference_type: Optional[str] = None
    metadata: Dict = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Report/Flag Models (for moderation)
class ContentReport(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    reporter_id: str
    content_type: str  # post, comment, user
    content_id: str
    reason: str  # spam, inappropriate, harassment, misinformation
    description: Optional[str] = None
    status: str = "pending"  # pending, reviewed, action_taken, dismissed
    reviewed_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# AI Models
class AIInteraction(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    image_id: str
    disease_prediction: str
    confidence_score: float
    suggested_treatment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class LearningAnalytics(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    crops_monitored: int = 0
    treatments_applied: int = 0
    success_rate: float = 0.0
    session_duration: int = 0
    interactions: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Recommendation(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    crop_id: Optional[str] = None
    treatment_id: Optional[str] = None
    reason: str
    priority: str = "medium"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Gamification Models
class Badge(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    description: str
    icon_url: Optional[str] = None
    criteria: dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserBadge(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    badge_id: str
    earned_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Leaderboard(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    score: int = 0
    rank: int = 0
    category: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Achievement(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    achievement_type: str
    description: str
    points: int = 0
    unlocked_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Supplier Models
class Supplier(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    location: str
    contact_info: dict = {}
    products_available: List[str] = []
    crop_association: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}



# Organic Solutions Models
class OrganicSolution(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description: str
    category: str
    success_rate: float = 0.0
    cost_per_acre: float = 0.0
    preparation_time: str
    ingredients: List[Dict] = []
    preparation_steps: List[str] = []
    application_method: str
    application_frequency: str
    diseases_treated: List[str] = []
    crops_suitable_for: List[str] = []
    precautions: List[str] = []
    local_names: Dict[str, str] = {}
    seasonal_effectiveness: Dict[str, str] = {}
    
    # NEW: Image fields
    image_url: Optional[str] = None  # Main solution image
    media_urls: List[str] = []  # Additional images/videos
    
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class SolutionApplication(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    solution_id: str
    user_id: str
    crop_id: Optional[str] = None
    photo_id: Optional[str] = None
    area_applied: Optional[float] = None  # in acres
    location: Optional[str] = None
    notes: Optional[str] = None
    before_photo_url: Optional[str] = None
    after_photo_url: Optional[str] = None
    status: str = "applied"  # applied, in_progress, completed
    outcome: Optional[str] = None  # success, partial, failed
    feedback: Optional[str] = None
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    follow_up_dates: List[datetime] = []

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SolutionRating(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    solution_id: str
    user_id: str
    rating: int  # 1-5
    review: Optional[str] = None
    effectiveness: Optional[int] = None  # 1-5
    ease_of_preparation: Optional[int] = None  # 1-5
    cost_effectiveness: Optional[int] = None  # 1-5
    would_recommend: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AnalysisFeedback(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    photo_id: str
    user_id: str
    rating: Optional[int] = None  # 1-5
    is_accurate: Optional[bool] = None
    comments: Optional[str] = None
    actual_disease: Optional[str] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Ingredient(BaseModel):
    name: str
    quantity: str
    local_name: str
    availability: str


class PreparationStep(BaseModel):
    step_number: int
    description: str
    duration: Optional[str] = None
    tips: Optional[str] = None


# Enhanced Treatment Model (if you want to update the existing one)
class EnhancedTreatment(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    crop_id: Optional[str] = None
    disease_or_pest_id: Optional[str] = None
    title: str
    description: str
    treatment_type: str  # organic, chemical, biological
    step_by_step_guide: List[str] = []
    ingredients_local_availability: List[Dict] = []
    estimated_time: str
    estimated_cost: Optional[float] = None
    success_rate: Optional[float] = None
    seasonal_relevance: List[str] = []
    media_urls: List[str] = []
    video_urls: List[str] = []
    precautions: List[str] = []
    local_names: Dict[str, str] = {}
    created_by: str
    verified_by: Optional[str] = None
    verification_status: str = "pending"  # pending, verified, rejected
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request/Response Models for API
class SolutionCreateRequest(BaseModel):
    title: str
    description: str
    category: str
    success_rate: Optional[float] = 0.0
    cost_per_acre: Optional[float] = 0.0
    preparation_time: str
    ingredients: List[Dict]
    preparation_steps: List[str]
    application_method: str
    application_frequency: str
    diseases_treated: List[str] = []
    crops_suitable_for: List[str] = []
    precautions: List[str] = []
    local_names: Optional[Dict[str, str]] = {}
    seasonal_effectiveness: Optional[Dict[str, str]] = {}


class SolutionUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    success_rate: Optional[float] = None
    cost_per_acre: Optional[float] = None
    preparation_time: Optional[str] = None
    ingredients: Optional[List[Dict]] = None
    preparation_steps: Optional[List[str]] = None
    application_method: Optional[str] = None
    application_frequency: Optional[str] = None
    diseases_treated: Optional[List[str]] = None
    crops_suitable_for: Optional[List[str]] = None
    precautions: Optional[List[str]] = None
    local_names: Optional[Dict[str, str]] = None
    seasonal_effectiveness: Optional[Dict[str, str]] = None


class ApplicationCreateRequest(BaseModel):
    crop_id: Optional[str] = None
    photo_id: Optional[str] = None
    area_applied: Optional[float] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    before_photo_url: Optional[str] = None


class ApplicationUpdateRequest(BaseModel):
    status: Optional[str] = None
    outcome: Optional[str] = None
    notes: Optional[str] = None
    after_photo_url: Optional[str] = None
    feedback: Optional[str] = None


class RatingCreateRequest(BaseModel):
    rating: int
    review: Optional[str] = None
    effectiveness: Optional[int] = None
    ease_of_preparation: Optional[int] = None
    cost_effectiveness: Optional[int] = None
    would_recommend: bool = True


class FeedbackCreateRequest(BaseModel):
    rating: Optional[int] = None
    is_accurate: Optional[bool] = None
    comments: Optional[str] = None
    actual_disease: Optional[str] = None

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


# Similarly, you can create request/response models for other entities as needed.
# Add these enhanced models to your models.py file

# Enhanced Seasonal/Advisory Models
class SeasonalCalendar(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    month: str  # January, February, etc.
    month_number: int  # 1-12
    season: str  # Kharif, Rabi, Summer/Zaid
    temperature_range: str  # "10-25°C", "Cool", etc.
    rainfall_pattern: Optional[str] = None  # "Heavy", "Moderate", "Low", "None"
    humidity_range: Optional[str] = None
    
    # General activities
    activities: List[str] = []
    
    # Crop-specific info
    crop_id: Optional[str] = None
    recommended_crops: List[str] = []
    
    # Actions and alerts
    preventive_actions: List[str] = []
    treatment_alerts: List[str] = []
    weather_alerts: List[str] = []
    
    # Agricultural operations
    sowing_activities: List[str] = []
    harvesting_activities: List[str] = []
    irrigation_guidelines: List[str] = []
    fertilization_schedule: List[str] = []
    
    # Pest and disease info
    common_pests: List[str] = []
    common_diseases: List[str] = []
    
    # Additional metadata
    region: Optional[str] = None  # For region-specific calendars
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SeasonInfo(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    season_name: str  # Kharif, Rabi, Summer/Zaid
    start_month: int  # 1-12
    end_month: int  # 1-12
    description: str
    characteristics: List[str] = []
    suitable_crops: List[str] = []
    climate_conditions: Dict[str, str] = {}  # {temperature, rainfall, humidity}
    challenges: List[str] = []
    opportunities: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CropCalendar(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    crop_id: str
    crop_name: str
    season: str  # Kharif, Rabi, Summer
    
    # Growth stages with timing
    growth_stages: List[Dict] = []  # [{stage_name, duration_days, activities}]
    
    # Month-wise activities
    monthly_activities: Dict[str, List[str]] = {}  # {month: [activities]}
    
    # Critical periods
    critical_irrigation_periods: List[str] = []
    critical_pest_periods: List[str] = []
    critical_disease_periods: List[str] = []
    
    # Timing information
    best_sowing_time: str
    expected_harvest_time: str
    total_duration_days: int
    
    # Requirements
    water_requirements: Dict[str, str] = {}  # {stage: requirement}
    nutrient_requirements: Dict[str, str] = {}
    
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class WeatherForecast(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    location: str
    district: str
    state: str
    
    # Forecast data
    forecast_date: datetime
    temperature_min: float
    temperature_max: float
    rainfall_mm: Optional[float] = None
    humidity_percent: Optional[float] = None
    wind_speed_kmph: Optional[float] = None
    
    # Conditions
    weather_condition: str  # "Clear", "Cloudy", "Rainy", etc.
    
    # Advisory based on forecast
    farming_advisory: List[str] = []
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AgriAdvisory(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    advisory_type: str  # weather, pest, disease, general, seasonal
    priority: str = "medium"  # low, medium, high, urgent
    
    # Content
    description: str
    recommended_actions: List[str] = []
    
    # Targeting
    target_crops: List[str] = []
    target_region: Optional[str] = None
    target_season: Optional[str] = None
    
    # Validity
    valid_from: datetime
    valid_until: datetime
    
    # Additional info
    source: Optional[str] = None  # Agricultural department, expert, etc.
    media_urls: List[str] = []
    
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class FarmingTip(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    category: str  # irrigation, fertilization, pest_control, etc.
    tip_text: str
    
    # Applicability
    applicable_months: List[str] = []
    applicable_seasons: List[str] = []
    applicable_crops: List[str] = []
    
    # Metadata
    difficulty_level: str = "medium"
    estimated_time: Optional[str] = None
    estimated_cost: Optional[str] = None
    
    # Engagement
    upvotes: int = 0
    downvotes: int = 0
    
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request/Response Models for API
class CalendarEntryRequest(BaseModel):
    month: str
    crop_id: Optional[str] = None
    activities: Optional[List[str]] = []
    preventive_actions: Optional[List[str]] = []
    treatment_alerts: Optional[List[str]] = []
    weather_alerts: Optional[List[str]] = []
    recommended_crops: Optional[List[str]] = []


class SeasonalAdvisoryRequest(BaseModel):
    title: str
    advisory_type: str
    priority: str = "medium"
    description: str
    recommended_actions: List[str]
    target_crops: Optional[List[str]] = []
    target_region: Optional[str] = None
    valid_from: datetime
    valid_until: datetime


class CropCalendarRequest(BaseModel):
    crop_id: str
    season: str
    growth_stages: List[Dict]
    best_sowing_time: str
    expected_harvest_time: str
    total_duration_days: int
    monthly_activities: Optional[Dict[str, List[str]]] = {}


class WeatherAlertRequest(BaseModel):
    location: str
    alert_type: str
    message: str
    recommended_action: str
    severity: str = "medium"  # low, medium, high
    crop_id: Optional[str] = None




# videotourtorials 


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
    video_url: Optional[str] = None  # For YouTube URLs
    youtube_url: Optional[str] = None  # NEW: YouTube video URL
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



# VAPI Models
class VAPIQuery(BaseModel):
    query: str
    context: Optional[Dict] = None
    user_location: Optional[str] = None

class VAPIFunctionCall(BaseModel):
    name: str
    parameters: Dict[str, Any]

class VAPIMessage(BaseModel):
    type: str
    functionCall: Optional[VAPIFunctionCall] = None
    call: Optional[Dict] = None

class VAPIRequest(BaseModel):
    message: VAPIMessage
class SuggestedAction(BaseModel):
    action: str
    label: str
    route: str

class VAPIResponse(BaseModel):
    response: str
    suggested_actions: List[SuggestedAction] = []
    relevant_data: Optional[Dict[str, Any]] = None

class WeatherAlert(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    location: str
    crop_id: Optional[str] = None
    alert_type: str  # rainfall, temperature, storm, humidity, wind, frost, heatwave, drought
    severity: str = "medium"  # low, medium, high, urgent
    message: str
    recommended_action: str
    valid_until: Optional[datetime] = None  # Alert expiry time
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional fields
    affected_crops: List[str] = []
    source: Optional[str] = None  # IMD, OpenWeatherMap, Manual, etc.
    acknowledged_by: List[str] = []  # User IDs who acknowledged
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# New WeatherForecast Model (for storing forecasts in DB)
class WeatherForecast(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    location: str
    district: str
    state: str
    
    # Forecast data
    forecast_date: datetime
    temperature_min: float
    temperature_max: float
    temperature_avg: Optional[float] = None
    rainfall_mm: Optional[float] = 0.0
    humidity_percent: Optional[float] = None
    wind_speed_kmph: Optional[float] = None
    
    # Conditions
    weather_condition: str  # Clear, Cloudy, Rainy, Stormy, etc.
    weather_description: Optional[str] = None
    
    # Advisory based on forecast
    farming_advisory: List[str] = []
    
    # Source tracking
    source: str = "OpenWeatherMap"
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Enhanced AdvisoryCalendar Model
class AdvisoryCalendar(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    crop_id: Optional[str] = None
    month: str
    month_number: Optional[int] = None  # 1-12
    season: Optional[str] = None  # Kharif, Rabi, Summer
    
    # Activities and advisories
    preventive_actions: List[str] = []
    treatment_alerts: List[str] = []
    weather_alerts: List[str] = []
    
    # Additional recommendations
    irrigation_schedule: Optional[List[str]] = []
    fertilization_schedule: Optional[List[str]] = []
    pest_watch: Optional[List[str]] = []
    
    # Metadata
    region: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Weather-based Advisory Model
class WeatherAdvisory(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    advisory_type: str  # weather, pest, disease, general, seasonal
    priority: str = "medium"  # low, medium, high, urgent
    
    # Content
    description: str
    recommended_actions: List[str] = []
    
    # Targeting
    target_crops: List[str] = []
    target_region: Optional[str] = None
    target_districts: List[str] = []
    target_season: Optional[str] = None
    
    # Weather conditions that triggered this advisory
    weather_conditions: Optional[Dict] = None  # {temp, rainfall, humidity, etc.}
    
    # Validity
    valid_from: datetime
    valid_until: datetime
    is_active: bool = True
    
    # Additional info
    source: Optional[str] = None
    media_urls: List[str] = []
    
    # Engagement
    views_count: int = 0
    helpful_count: int = 0
    
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# User Weather Preferences
class WeatherPreferences(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    
    # Notification settings
    enable_weather_alerts: bool = True
    enable_rainfall_alerts: bool = True
    enable_temperature_alerts: bool = True
    enable_storm_alerts: bool = True
    
    # Alert thresholds
    high_temp_threshold: Optional[float] = 35.0  # Celsius
    low_temp_threshold: Optional[float] = 15.0
    heavy_rain_threshold: Optional[float] = 50.0  # mm
    
    # Preferred alert times
    morning_alert: bool = True
    evening_alert: bool = True
    
    # Location preferences
    primary_location: Optional[str] = None
    additional_locations: List[str] = []
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request Models for Weather Endpoints
class WeatherAlertCreateRequest(BaseModel):
    location: str
    crop_id: Optional[str] = None
    alert_type: str
    severity: str = "medium"
    message: str
    recommended_action: str
    valid_until: Optional[str] = None  # ISO format datetime string
    affected_crops: Optional[List[str]] = []


class WeatherAdvisoryCreateRequest(BaseModel):
    title: str
    advisory_type: str
    priority: str = "medium"
    description: str
    recommended_actions: List[str]
    target_crops: Optional[List[str]] = []
    target_region: Optional[str] = None
    target_districts: Optional[List[str]] = []
    valid_from: str  # ISO format datetime
    valid_until: str  # ISO format datetime
    weather_conditions: Optional[Dict] = None


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


# Consultation Session Models
class ConsultationSession(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    farmer_id: str
    farmer_name: str
    specialist_id: Optional[str] = None
    specialist_name: Optional[str] = None
    
    # Session details
    session_type: str  # video, audio, chat
    status: str = "pending"  # pending, active, completed, cancelled
    topic: Optional[str] = None
    description: Optional[str] = None
    
    # Crop/Disease context
    related_crop_photo_id: Optional[str] = None
    related_disease: Optional[str] = None
    
    # Scheduling
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    
    # WebRTC/Connection details
    room_id: Optional[str] = None
    webrtc_offer: Optional[Dict] = None
    webrtc_answer: Optional[Dict] = None
    ice_candidates: List[Dict] = []
    
    # Session data
    messages: List[str] = []  # Message IDs
    shared_photos: List[str] = []  # Photo URLs
    notes: Optional[str] = None
    
    # Feedback
    farmer_rating: Optional[int] = None
    farmer_feedback: Optional[str] = None
    specialist_notes: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ConsultationMessage(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    session_id: str
    sender_id: str
    sender_name: str
    sender_role: str  # farmer, specialist
    
    # Message content
    message_type: str = "text"  # text, image, file, system, diagnosis, prescription
    message_text: Optional[str] = None
    media_url: Optional[str] = None
    
    # Metadata
    read: bool = False
    read_at: Optional[datetime] = None
    
    # Special message types
    diagnosis_data: Optional[Dict] = None  # For diagnosis messages
    prescription_data: Optional[Dict] = None  # For treatment recommendations
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SpecialistAvailability(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    specialist_id: str
    
    # Availability schedule
    day_of_week: int  # 0-6 (Monday-Sunday)
    start_time: str  # "09:00"
    end_time: str  # "17:00"
    
    # Status
    is_available: bool = True
    max_concurrent_sessions: int = 3
    
    # Breaks/Unavailability
    break_start: Optional[str] = None
    break_end: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SpecialistProfile(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    
    # Professional info
    specialization: List[str] = []  # crop_diseases, pests, organic_farming, etc.
    experience_years: int = 0
    qualification: Optional[str] = None
    languages: List[str] = ["telugu", "english"]
    
    # Expertise areas
    crops_expertise: List[str] = []
    diseases_expertise: List[str] = []
    
    # Ratings and stats
    average_rating: float = 0.0
    total_consultations: int = 0
    total_ratings: int = 0
    
    # Availability
    is_online: bool = False
    last_active: Optional[datetime] = None
    consultation_fee: Optional[float] = None  # 0 for free
    
    # Bio
    bio: Optional[str] = None
    profile_image: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ConsultationRequest(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    farmer_id: str
    farmer_name: str
    
    # Request details
    preferred_specialist_id: Optional[str] = None
    session_type: str  # video, audio, chat
    topic: str
    description: Optional[str] = None
    urgency: str = "normal"  # low, normal, high, urgent
    
    # Context
    related_crop_photo_id: Optional[str] = None
    related_disease: Optional[str] = None
    
    # Status
    status: str = "pending"  # pending, accepted, rejected, expired
    accepted_by_specialist_id: Optional[str] = None
    
    # Scheduling preferences
    preferred_datetime: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request/Response Models
class ConsultationSessionCreateRequest(BaseModel):
    session_type: str
    topic: str
    description: Optional[str] = None
    related_crop_photo_id: Optional[str] = None
    scheduled_at: Optional[str] = None  # ISO datetime
    preferred_specialist_id: Optional[str] = None


class ConsultationMessageRequest(BaseModel):
    message_type: str = "text"
    message_text: Optional[str] = None
    media_url: Optional[str] = None
    diagnosis_data: Optional[Dict] = None
    prescription_data: Optional[Dict] = None


class WebRTCSignalRequest(BaseModel):
    signal_type: str  # offer, answer, ice_candidate
    signal_data: Dict


class ConsultationFeedbackRequest(BaseModel):
    rating: int  # 1-5
    feedback: Optional[str] = None


class SpecialistAvailabilityRequest(BaseModel):
    day_of_week: int
    start_time: str
    end_time: str
    is_available: bool = True
    break_start: Optional[str] = None
    break_end: Optional[str] = None

# Add these Supplier models to your models.py file

class Supplier(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    location: str
    contact_info: dict = {}  # {phone, email, address, website}
    products_available: List[str] = []
    crop_association: List[str] = []
    rating: float = 0.0
    total_ratings: int = 0
    verified: bool = False
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SupplierReview(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    supplier_id: str
    user_id: str
    user_name: str
    rating: int  # 1-5
    review: Optional[str] = None
    helpful_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SupplierInquiry(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    supplier_id: str
    user_id: str
    user_name: str
    user_phone: Optional[str] = None
    product_interest: List[str] = []
    message: Optional[str] = None
    status: str = "pending"  # pending, contacted, completed, cancelled
    created_at: datetime = Field(default_factory=datetime.utcnow)
    responded_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request Models for Supplier APIs
class SupplierCreateRequest(BaseModel):
    name: str
    location: str
    phone: str
    email: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None
    products_available: List[str]
    crop_association: Optional[List[str]] = []


class SupplierUpdateRequest(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None
    products_available: Optional[List[str]] = None
    crop_association: Optional[List[str]] = None


class SupplierReviewRequest(BaseModel):
    rating: int  # 1-5
    review: Optional[str] = None


class SupplierInquiryRequest(BaseModel):
    product_interest: List[str]
    message: Optional[str] = None

class ImpactStats(BaseModel):
    total_crops_monitored: int
    treatments_applied: int
    success_rate: float
    cost_saved: float
    chemical_reduction: float
    organic_solutions_adopted: int
    community_contributions: int
    badges_earned: int
    streak_days: int

class MonthlyProgress(BaseModel):
    organic_solutions_progress: float
    crop_health_improvement: float
    community_engagement: float

class CostSavings(BaseModel):
    total_saved: float
    breakdown: Dict[str, float]

class EnvironmentalImpact(BaseModel):
    chemical_reduction_kg: float
    chemical_reduction_percentage: float
    water_saved_liters: float
    carbon_footprint_reduced_kg: float


    # product 


# Add these models to your models.py file

class Product(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    seller_id: str
    seller_name: str
    
    # Product details
    title: str
    description: str
    category: str  # seeds, fertilizers, tools, pesticides, equipment, produce
    subcategory: Optional[str] = None
    
    # Pricing
    price: float
    unit: str  # kg, liter, piece, packet, bag
    min_order_quantity: Optional[float] = 1.0
    max_order_quantity: Optional[float] = None
    
    # Stock
    stock_available: float
    stock_unit: str
    
    # Product specifications
    organic_certified: bool = False
    brand: Optional[str] = None
    specifications: Dict[str, Any] = {}
    
    # Media
    images: List[str] = []
    thumbnail: Optional[str] = None
    
    # Location
    location: str
    district: str
    state: str = "Andhra Pradesh"
    
    # Crops applicable
    suitable_for_crops: List[str] = []
    
    # Status
    status: str = "active"  # active, sold_out, inactive, archived
    is_featured: bool = False
    
    # Engagement
    views_count: int = 0
    inquiries_count: int = 0
    orders_count: int = 0
    rating: float = 0.0
    total_ratings: int = 0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ProductOrder(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    order_number: str  # Unique order ID
    
    # Buyer info
    buyer_id: str
    buyer_name: str
    buyer_phone: str
    buyer_email: Optional[str] = None
    
    # Seller info
    seller_id: str
    seller_name: str
    
    # Product info
    product_id: str
    product_title: str
    product_image: Optional[str] = None
    
    # Order details
    quantity: float
    unit: str
    unit_price: float
    total_price: float
    
    # Shipping address
    shipping_address: Dict[str, str] = {}  # {name, phone, address_line1, address_line2, city, district, state, pincode}
    
    # Payment
    payment_status: str = "pending"  # pending, processing, completed, failed, refunded
    payment_method: str = "stripe"
    stripe_payment_intent_id: Optional[str] = None
    stripe_charge_id: Optional[str] = None
    payment_completed_at: Optional[datetime] = None
    
    # Order status
    order_status: str = "pending"  # pending, confirmed, preparing, shipped, delivered, cancelled
    
    # Tracking
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    
    # Notes
    buyer_notes: Optional[str] = None
    seller_notes: Optional[str] = None
    cancellation_reason: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ProductReview(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    product_id: str
    order_id: str
    user_id: str
    user_name: str
    
    rating: int  # 1-5
    review: Optional[str] = None
    
    # Specific ratings
    quality_rating: Optional[int] = None
    value_rating: Optional[int] = None
    delivery_rating: Optional[int] = None
    
    images: List[str] = []
    
    helpful_count: int = 0
    verified_purchase: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request Models
class ProductCreateRequest(BaseModel):
    title: str
    description: str
    category: str
    subcategory: Optional[str] = None
    price: float
    unit: str
    min_order_quantity: Optional[float] = 1.0
    stock_available: float
    stock_unit: str
    organic_certified: bool = False
    brand: Optional[str] = None
    specifications: Optional[Dict[str, Any]] = {}
    images: List[str] = []
    location: str
    district: str
    suitable_for_crops: Optional[List[str]] = []


class ProductUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    stock_available: Optional[float] = None
    status: Optional[str] = None
    images: Optional[List[str]] = None


class OrderCreateRequest(BaseModel):
    product_id: str
    quantity: float
    shipping_address: Dict[str, str]
    buyer_notes: Optional[str] = None


class OrderUpdateRequest(BaseModel):
    order_status: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    seller_notes: Optional[str] = None


class ProductReviewRequest(BaseModel):
    rating: int
    review: Optional[str] = None
    quality_rating: Optional[int] = None
    value_rating: Optional[int] = None
    delivery_rating: Optional[int] = None
    images: Optional[List[str]] = []


# chat bot mmode l

# Add these models to your models.py file

class ChatbotConversation(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    user_message: str
    bot_response: str
    intent: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ChatbotFeedback(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    conversation_id: str
    user_id: str
    helpful: bool
    comment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ChatbotMessageRequest(BaseModel):
    message: str


class ChatbotFeedbackRequest(BaseModel):
    conversation_id: str
    helpful: bool
    comment: Optional[str] = None