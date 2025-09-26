# backend/python/fitness_api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class PlanReq(BaseModel):
    uid: Optional[str] = None
    age: int
    height: int
    weight: int
    goal: str

@app.post("/plan")
def plan(req: PlanReq):
    bmi = req.weight / ((req.height/100)**2)
    plan = {"weeks":4, "goal": req.goal, "weekly": []}
    if req.goal == 'fat_loss':
        plan['weekly'] = [
            {"day":1,"workouts":[{"name":"Circuit A","sets":3,"reps":"12"}]},
            {"day":2,"workouts":[{"name":"LISS 30min","sets":1,"reps":""}]}
        ]
    else:
        plan['weekly'] = [
            {"day":1,"workouts":[{"name":"Strength Lower","sets":4,"reps":"6-8"}]},
            {"day":2,"workouts":[{"name":"Strength Upper","sets":4,"reps":"6-8"}]}
        ]
    plan['bmi'] = round(bmi,1)
    return plan
