# backend/routers/fitness_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, conint
from typing import List, Optional, Dict, Any
import pandas as pd
import os, sqlite3, json, math, datetime
from auth import verify_firebase_token
from utils import MODELS  # reuse MODELS dict to get model dir if needed
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# --- DB (simple sqlite) ---
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fitness.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            uid TEXT PRIMARY KEY,
            age INTEGER,
            sex TEXT,
            height_cm REAL,
            weight_kg REAL,
            body_type TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS workout_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT,
            date TEXT,
            exercise TEXT,
            sets INTEGER,
            reps INTEGER,
            notes TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_plans (
            uid TEXT,
            plan_json TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- Load exercise dataset ---
EXERCISES_CSV = os.path.join(os.path.dirname(__file__), "..", "models", "fitness_data.csv")
if not os.path.exists(EXERCISES_CSV):
    logger.warning("fitness_data.csv not found at %s", EXERCISES_CSV)
fitness_df = pd.read_csv(EXERCISES_CSV) if os.path.exists(EXERCISES_CSV) else pd.DataFrame()

# --- Pydantic models ---
class FitnessRequest(BaseModel):
    age: conint(ge=10, le=100)
    sex: Optional[str] = Field(None, description="M/F/other")
    height_cm: float
    weight_kg: float
    goal: str = Field(..., description="strength | weight_loss | flexibility | core | endurance")
    level: str = Field("beginner", description="beginner | intermediate | advanced")
    equipment: List[str] = Field(default_factory=list, description="e.g. ['dumbbell','barbell','none']")
    days_per_week: conint(ge=1, le=7) = 3
    time_per_session_min: conint(ge=10, le=180) = 45
    injuries: Optional[List[str]] = None
    weeks: conint(ge=1, le=12) = 4

class ExerciseItem(BaseModel):
    exercise: str
    sets: int
    reps: str
    rest_sec: int
    notes: Optional[str] = None
    video_url: Optional[str] = None
    body_part: Optional[str] = None

class SessionPlan(BaseModel):
    day: int
    title: str
    exercises: List[ExerciseItem]

class WeeklyPlan(BaseModel):
    week: int
    sessions: List[SessionPlan]

class FullPlanResponse(BaseModel):
    uid: Optional[str]
    bmi: float
    bmi_category: str
    weeks: List[WeeklyPlan]

# --- Helpers ---
def compute_bmi(weight_kg: float, height_cm: float):
    h = height_cm / 100.0
    if h <= 0:
        raise ValueError("Invalid height")
    bmi = weight_kg / (h*h)
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Normal"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    return round(bmi, 1), cat

def filter_exercises(goal: str, level: str, equipment: List[str], injuries: Optional[List[str]] = None):
    if fitness_df.empty:
        return []
    df = fitness_df.copy()
    # normalize columns if needed
    df['primary_goal'] = df['primary_goal'].astype(str).str.lower()
    df['secondary_goals'] = df['secondary_goals'].astype(str).str.lower()
    df['level'] = df['level'].astype(str).str.lower()
    df['equipment'] = df['equipment'].astype(str).str.lower()
    df['body_part'] = df['body_part'].astype(str).str.lower()

    goal = goal.lower()
    level = level.lower()
    equipment = [e.lower() for e in equipment]

    # filter by goal
    df = df[df['primary_goal'].str.contains(goal) | df['secondary_goals'].str.contains(goal)]

    # filter by level (or 'all')
    df = df[(df['level'] == level) | (df['level'] == 'all')]

    # equipment filter: allow exercise if its equipment is 'none' or user has it
    def equip_ok(row):
        req = row['equipment']
        if req in ('none', '', 'bodyweight'):
            return True
        # if multiple equipment separated by '|' check if any is available
        reqs = [x.strip() for x in req.split('|')]
        for r in reqs:
            if r in equipment:
                return True
        return False

    df = df[df.apply(equip_ok, axis=1)]

    # injuries: basic string matching exclusion (if injury word in notes -> exclude)
    if injuries:
        inj_lower = [x.lower() for x in injuries]
        def inj_ok(row):
            notes = str(row.get('notes','')).lower()
            for inj in inj_lower:
                if inj in notes:
                    return False
            return True
        df = df[df.apply(inj_ok, axis=1)]

    return df

def choose_exercises_for_session(candidates_df: pd.DataFrame, body_parts: List[str], n_ex: int):
    # try to choose exercises that cover given body parts; fallback otherwise
    chosen = []
    for part in body_parts:
        subset = candidates_df[candidates_df['body_part'].str.contains(part, na=False)]
        if not subset.empty:
            chosen.append(subset.sample(1).iloc[0].to_dict())
    # fill remaining randomly
    if len(chosen) < n_ex:
        remaining = candidates_df.sample(min(n_ex - len(chosen), len(candidates_df)))
        for _, r in remaining.iterrows():
            chosen.append(r.to_dict())
    return chosen[:n_ex]

def sets_reps_for_goal(goal:str, level:str):
    goal = goal.lower()
    level = level.lower()
    if goal == "strength":
        if level == "beginner": return (3, "5")
        if level == "intermediate": return (4, "4-6")
        return (4, "3-5")
    if goal == "weight_loss" or goal == "endurance":
        if level == "beginner": return (2, "12-15")
        if level == "intermediate": return (3, "12-20")
        return (3, "15-20")
    # default hypertrophy / general fitness
    if level == "beginner": return (2, "8-12")
    if level == "intermediate": return (3, "8-12")
    return (3, "8-12")

def generate_weekly_plan(req: FitnessRequest):
    bmi, bmi_cat = compute_bmi(req.weight_kg, req.height_cm)
    candidates = filter_exercises(req.goal, req.level, req.equipment, req.injuries)
    if candidates.empty:
        raise HTTPException(status_code=404, detail="No exercises available for the given filters")

    # decide split
    days = req.days_per_week
    weeks = []
    # simple mapping of splits
    if days <= 3:
        split = [["full"]]*days  # all full body
    elif days == 4:
        split = [["upper"], ["lower"], ["upper"], ["lower"]]
    elif days == 5:
        split = [["push"], ["pull"], ["legs"], ["core"], ["cardio"]]
    else:
        split = [["full"]]*days

    # body part mapping for named sessions
    mapping = {
        "full": ["fullbody"],
        "upper": ["chest","back","shoulders","arms"],
        "lower": ["legs","glutes"],
        "push": ["chest","shoulders","triceps"],
        "pull": ["back","biceps"],
        "legs": ["legs","glutes"],
        "core": ["core"],
        "cardio": ["fullbody"]
    }

    # create plan for each week and session
    for w in range(1, req.weeks + 1):
        sessions = []
        for day_idx, session_name_list in enumerate(split, start=1):
            # session_name_list is list with a single name like ["upper"]
            name = session_name_list[0]
            body_parts = mapping.get(name, ["fullbody"])
            # number of exercises scaled by time_per_session
            n_ex = 5 if req.time_per_session_min >= 30 else 3
            chosen = choose_exercises_for_session(candidates, body_parts, n_ex)
            exercises_items = []
            for ex in chosen:
                sets, reps = sets_reps_for_goal(req.goal, req.level)
                # progressive overload: increase reps/sets slightly each week
                # simple rule: add 1 rep every two weeks
                extra_reps = math.floor((w-1)/2)
                # reps might be range "8-12" -> we append +extra to both ends naive approach
                reps_str = str(reps)
                try:
                    if "-" in reps_str:
                        low, high = [int(x) for x in reps_str.split("-")]
                        low += extra_reps
                        high += extra_reps
                        reps_out = f"{low}-{high}"
                    else:
                        reps_out = str(int(reps_str) + extra_reps)
                except:
                    reps_out = reps_str

                rest = 60 if req.goal in ("strength","hypertrophy") else 45

                exercises_items.append({
                    "exercise": ex.get("exercise"),
                    "sets": sets,
                    "reps": reps_out,
                    "rest_sec": rest,
                    "notes": ex.get("notes"),
                    "video_url": ex.get("video_url"),
                    "body_part": ex.get("body_part")
                })

            sessions.append({
                "day": day_idx,
                "title": f"Week {w} - {name.capitalize()}",
                "exercises": exercises_items
            })

        weeks.append({
            "week": w,
            "sessions": sessions
        })

    return {
        "bmi": bmi,
        "bmi_category": bmi_cat,
        "weeks": weeks
    }

# --- Endpoints ---
@router.post("/recommend", response_model=FullPlanResponse)
def recommend_fitness(req: FitnessRequest, user=Depends(verify_firebase_token)):
    uid = user.get("uid") if isinstance(user, dict) else None
    try:
        plan = generate_weekly_plan(req)
        # persist plan
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO user_plans (uid, plan_json, created_at) VALUES (?, ?, ?)",
                  (uid, json.dumps(plan), datetime.datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        return {"uid": uid, "bmi": plan["bmi"], "bmi_category": plan["bmi_category"], "weeks": plan["weeks"]}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error generating plan")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/exercises")
def list_exercises(goal: Optional[str]=None, level: Optional[str]=None, equipment: Optional[str]=None, user=Depends(verify_firebase_token)):
    if fitness_df.empty:
        return {"exercises": []}
    df = fitness_df.copy()
    if goal:
        df = df[df['primary_goal'].str.contains(goal, case=False, na=False) | df['secondary_goals'].str.contains(goal, case=False, na=False)]
    if level:
        df = df[(df['level'].str.lower()==level.lower()) | (df['level'].str.lower()=='all')]
    if equipment:
        df = df[df['equipment'].str.contains(equipment, case=False, na=False)]
    return {"exercises": df.to_dict(orient="records")}

@router.post("/log")
def log_workout(uid: str, exercise: str, sets: int, reps: str, notes: Optional[str]=None, user=Depends(verify_firebase_token)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO workout_logs (uid, date, exercise, sets, reps, notes) VALUES (?, ?, ?, ?, ?, ?)",
              (uid, datetime.datetime.utcnow().isoformat(), exercise, sets, reps, notes))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@router.get("/progress")
def get_progress(uid: str, user=Depends(verify_firebase_token)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, exercise, sets, reps, notes FROM workout_logs WHERE uid = ? ORDER BY date DESC LIMIT 100", (uid,))
    rows = c.fetchall()
    conn.close()
    logs = [{"date": r[0], "exercise": r[1], "sets": r[2], "reps": r[3], "notes": r[4]} for r in rows]
    return {"logs": logs}
