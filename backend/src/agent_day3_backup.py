import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

WELLNESS_LOG_FILE = "wellness_log.json"

# Current session state
current_session = {
    "mood": None,
    "energy": None,
    "stress_factors": [],
    "objectives": [],
    "timestamp": None,
    "summary": None
}

def load_wellness_history() -> List[Dict[str, Any]]:
    """Load previous wellness check-ins from JSON file"""
    if not os.path.exists(WELLNESS_LOG_FILE):
        return []
    
    try:
        with open(WELLNESS_LOG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading wellness history: {e}")
        return []

def save_wellness_entry(entry: Dict[str, Any]):
    """Save a wellness check-in entry to JSON file"""
    history = load_wellness_history()
    history.append(entry)
    
    with open(WELLNESS_LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Saved wellness entry: {entry['timestamp']}")

def get_last_checkin() -> Dict[str, Any] | None:
    """Get the most recent check-in"""
    history = load_wellness_history()
    return history[-1] if history else None

def format_history_context() -> str:
    """Format previous check-ins for context"""
    last = get_last_checkin()
    if not last:
        return "This is your first check-in with me."
    
    last_date = last.get('timestamp', 'last time')
    last_mood = last.get('mood', 'not specified')
    last_objectives = last.get('objectives', [])
    
    context = f"Last check-in was on {last_date}. You mentioned feeling {last_mood}"
    if last_objectives:
        context += f" and wanted to work on: {', '.join(last_objectives[:2])}"
    
    return context

def reset_session():
    """Reset current session state"""
    global current_session
    current_session = {
        "mood": None,
        "energy": None,
        "stress_factors": [],
        "objectives": [],
        "timestamp": datetime.now().isoformat(),
        "summary": None
    }


class WellnessCompanion(Agent):
    def __init__(self) -> None:
        history_context = format_history_context()
        
        super().__init__(
            instructions=f"""You are a supportive Health & Wellness Voice Companion. Your role is to conduct brief daily check-ins with users to support their wellbeing.

IMPORTANT CONTEXT:
{history_context}

YOUR APPROACH:
- You are warm, empathetic, and grounded
- You ask open-ended questions and listen actively
- You NEVER diagnose, prescribe, or give medical advice
- You offer small, actionable suggestions only
- You keep responses concise for voice interaction

CHECK-IN FLOW:
1. Greet warmly and reference previous check-in if available
2. Ask about mood and energy (one question at a time)
3. Gently explore any stress factors
4. Ask about 1-3 intentions/objectives for the day
5. Offer ONE small, practical suggestion based on what they shared
6. Summarize their mood and objectives
7. Use the save_checkin tool to persist the conversation

EXAMPLE QUESTIONS:
- "How are you feeling today?"
- "What's your energy like right now?"
- "Is anything weighing on your mind?"
- "What would you like to accomplish today?"
- "What's one thing you want to do for yourself today?"

ADVICE STYLE (keep it simple):
- Break big goals into smaller steps
- Suggest short breaks or walks
- Encourage self-compassion
- Remind them to stay hydrated
- Simple grounding techniques (deep breaths, stretching)

REMEMBER:
- One question at a time
- No medical claims
- Stay supportive, never judgmental
- Keep it conversational and natural
- No emojis or special formatting in voice responses
- Use tools to track mood, objectives, and save the check-in""",
        )

    @function_tool
    async def record_mood(
        self,
        context: RunContext,
        mood_description: str,
        energy_level: str = "not specified"
    ):
        """Record the user's current mood and energy level.
        
        Args:
            mood_description: How the user describes their mood (e.g., "good", "tired", "stressed", "energized")
            energy_level: User's energy level (e.g., "high", "medium", "low", "exhausted")
        """
        global current_session
        current_session["mood"] = mood_description
        current_session["energy"] = energy_level
        
        logger.info(f"Recorded mood: {mood_description}, energy: {energy_level}")
        return f"Noted: Mood is {mood_description}, energy level is {energy_level}"

    @function_tool
    async def record_stress_factor(
        self,
        context: RunContext,
        stress_factor: str
    ):
        """Record something that's stressing the user or on their mind.
        
        Args:
            stress_factor: What's causing stress or concern
        """
        global current_session
        current_session["stress_factors"].append(stress_factor)
        
        logger.info(f"Recorded stress factor: {stress_factor}")
        return f"I hear you - {stress_factor} is on your mind"

    @function_tool
    async def record_objective(
        self,
        context: RunContext,
        objective: str
    ):
        """Record a goal or intention the user has for the day.
        
        Args:
            objective: What the user wants to accomplish or focus on today
        """
        global current_session
        current_session["objectives"].append(objective)
        
        logger.info(f"Recorded objective: {objective}")
        return f"Added to your objectives: {objective}"

    @function_tool
    async def save_checkin(
        self,
        context: RunContext,
        summary: str
    ):
        """Save the complete check-in to the wellness log.
        
        Args:
            summary: A brief one-sentence summary of the check-in
        """
        global current_session
        
        if not current_session["mood"]:
            return "Cannot save check-in yet - mood hasn't been recorded"
        
        if not current_session["objectives"]:
            return "Cannot save check-in yet - no objectives recorded"
        
        current_session["summary"] = summary
        current_session["timestamp"] = datetime.now().isoformat()
        
        # Save to file
        save_wellness_entry(current_session.copy())
        
        # Format response
        objectives_text = "\n".join([f"  • {obj}" for obj in current_session["objectives"]])
        stress_text = ", ".join(current_session["stress_factors"]) if current_session["stress_factors"] else "nothing specific"
        
        response = f"""Check-in saved successfully!

Today's Summary:
• Mood: {current_session['mood']}
• Energy: {current_session['energy']}
• On your mind: {stress_text}
• Your objectives:
{objectives_text}

{summary}

I'm here whenever you need to check in. Take care!"""
        
        # Reset for next session
        reset_session()
        
        return response

    @function_tool
    async def review_recent_checkins(
        self,
        context: RunContext,
        days: int = 7
    ):
        """Review recent check-ins to provide insights.
        
        Args:
            days: Number of recent days to review (default: 7)
        """
        history = load_wellness_history()
        
        if not history:
            return "No previous check-ins found yet."
        
        recent = history[-min(days, len(history)):]
        
        # Simple analysis
        moods = [entry.get('mood', 'unknown') for entry in recent]
        total_objectives = sum(len(entry.get('objectives', [])) for entry in recent)
        
        response = f"""Here's a look at your last {len(recent)} check-in(s):

Recent moods: {', '.join(moods[-5:])}
Total objectives set: {total_objectives}
Check-ins completed: {len(recent)}

"""
        
        if len(recent) >= 3:
            response += "You've been consistent with checking in - that's great for self-awareness!"
        
        return response


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Reset session state for new connection
    reset_session()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=WellnessCompanion(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))