import logging
import json
import os
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

CONTENT_FILE = "shared-data/day4_tutor_content.json"

# Global session state
session_state = {
    "current_mode": "coordinator",
    "current_concept": None,
    "concepts": [],
    "session": None
}

def load_tutor_content() -> List[Dict[str, Any]]:
    """Load tutor content from JSON file"""
    if not os.path.exists(CONTENT_FILE):
        logger.error(f"Content file not found: {CONTENT_FILE}")
        return []
    
    try:
        with open(CONTENT_FILE, "r") as f:
            content = json.load(f)
            logger.info(f"Loaded {len(content)} concepts from {CONTENT_FILE}")
            return content
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        return []

def get_concept_by_id(concept_id: str) -> Dict[str, Any] | None:
    """Get a concept by ID"""
    concepts = session_state.get("concepts", [])
    for concept in concepts:
        if concept["id"].lower() == concept_id.lower():
            return concept
    return None

def get_concept_by_title(title: str) -> Dict[str, Any] | None:
    """Get a concept by title"""
    concepts = session_state.get("concepts", [])
    for concept in concepts:
        if concept["title"].lower() == title.lower():
            return concept
    return None

def get_available_concepts() -> str:
    """Get formatted list of available concepts"""
    concepts = session_state.get("concepts", [])
    if not concepts:
        return "No concepts loaded"
    
    concept_list = [c["title"] for c in concepts]
    return f"Available topics: {', '.join(concept_list)}"


class UnifiedTutorAgent(Agent):
    """Unified agent that handles all three modes with voice switching"""
    
    def __init__(self) -> None:
        concepts_info = get_available_concepts()
        
        super().__init__(
            instructions=f"""You are an interactive Teach-the-Tutor learning system with THREE learning modes.

AVAILABLE TOPICS:
{concepts_info}

THREE LEARNING MODES:

1. LEARN MODE (You are Matthew, the Teacher)
   - Explain concepts clearly and thoroughly
   - Use examples and analogies
   - Break down complex ideas
   - Encourage questions

2. QUIZ MODE (You are Alicia, the Quiz Master)
   - Ask questions to test understanding
   - Give encouraging feedback
   - Celebrate correct answers
   - Gently guide on wrong answers

3. TEACH BACK MODE (You are Ken, the Listener)
   - Ask the user to explain concepts to you
   - Listen carefully to their explanation
   - Give constructive, specific feedback
   - Highlight what they did well
   - Point out what they missed

IMPORTANT BEHAVIOR:
- When user first connects, greet them and explain the three modes
- Ask which mode they'd like to start with
- Use the switch_to_learn, switch_to_quiz, or switch_to_teachback tools to change modes
- After switching, adopt that mode's personality and voice style
- Users can switch modes anytime by asking

CURRENT STATUS:
- Mode: {session_state.get('current_mode', 'coordinator')}

REMEMBER: 
- Keep responses conversational for voice
- One question or instruction at a time
- Be encouraging and supportive""",
        )

    @function_tool
    async def switch_to_learn(
        self,
        context: RunContext,
        topic: str
    ):
        """Switch to LEARN mode where you explain concepts (Matthew's voice).
        
        Args:
            topic: The topic to teach (e.g., "variables", "loops", "functions")
        """
        # Try to find concept by ID or title
        concept = get_concept_by_id(topic) or get_concept_by_title(topic)
        
        if not concept:
            available = get_available_concepts()
            return f"I don't have that topic. {available}"
        
        session_state["current_mode"] = "learn"
        session_state["current_concept"] = concept
        
        logger.info(f"Switched to LEARN mode for {concept['title']}")
        
        # Return the explanation
        return f"""Let me teach you about {concept['title']}.

{concept['summary']}

Do you have any questions about this? Or would you like to hear more examples?"""

    @function_tool
    async def switch_to_quiz(
        self,
        context: RunContext,
        topic: str
    ):
        """Switch to QUIZ mode where you ask questions (Alicia's voice).
        
        Args:
            topic: The topic to quiz on (e.g., "variables", "loops", "functions")
        """
        concept = get_concept_by_id(topic) or get_concept_by_title(topic)
        
        if not concept:
            available = get_available_concepts()
            return f"I don't have that topic. {available}"
        
        session_state["current_mode"] = "quiz"
        session_state["current_concept"] = concept
        
        logger.info(f"Switched to QUIZ mode for {concept['title']}")
        
        return f"""Ready to test your knowledge on {concept['title']}?

Here's your question: {concept['sample_question']}

Take your time!"""

    @function_tool
    async def switch_to_teachback(
        self,
        context: RunContext,
        topic: str
    ):
        """Switch to TEACH BACK mode where the user explains concepts (Ken's voice).
        
        Args:
            topic: The topic for teach-back (e.g., "variables", "loops", "functions")
        """
        concept = get_concept_by_id(topic) or get_concept_by_title(topic)
        
        if not concept:
            available = get_available_concepts()
            return f"I don't have that topic. {available}"
        
        session_state["current_mode"] = "teach_back"
        session_state["current_concept"] = concept
        
        logger.info(f"Switched to TEACH BACK mode for {concept['title']}")
        
        return f"""Excellent! Now it's your turn to be the teacher.

Explain {concept['title']} to me as if I know nothing about it. Take your time and teach me everything you know!"""

    @function_tool
    async def provide_feedback(
        self,
        context: RunContext,
        strengths: str,
        improvements: str,
        overall: str = "Good"
    ):
        """Provide detailed feedback on a teach-back explanation (only use in TEACH BACK mode).
        
        Args:
            strengths: What the user explained well
            improvements: What they could improve or missed
            overall: Overall assessment (Excellent/Good/Needs Work)
        """
        if session_state.get("current_mode") != "teach_back":
            return "Feedback is only available in TEACH BACK mode"
        
        concept = session_state.get("current_concept")
        concept_name = concept["title"] if concept else "this topic"
        
        feedback = f"""Overall Assessment: {overall}

What you did well:
{strengths}

Areas to strengthen:
{improvements}

Great effort! Teaching is one of the best ways to learn. Want to try another topic or switch modes?"""
        
        return feedback

    @function_tool
    async def list_topics(self, context: RunContext):
        """List all available topics/concepts"""
        concepts = session_state.get("concepts", [])
        if not concepts:
            return "No topics available"
        
        topics = "\n".join([f"• {c['title']} - {c['summary'][:50]}..." for c in concepts])
        return f"Available topics:\n{topics}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Load tutor content
    session_state["concepts"] = load_tutor_content()
    
    if not session_state["concepts"]:
        logger.error("No concepts loaded! Check if content file exists.")
    else:
        logger.info(f"Successfully loaded {len(session_state['concepts'])} concepts")

    # Default to coordinator mode with Matthew voice
    voice = "en-US-matthew"
    current_mode = session_state.get("current_mode", "coordinator")
    
    # Change voice based on mode (this would need voice switching in a real handoff implementation)
    # For now, we'll use Matthew as default
    if current_mode == "quiz":
        voice = "en-US-alicia"
    elif current_mode == "teach_back":
        voice = "en-US-ken"

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=voice, 
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

    # Store session for potential handoffs
    session_state["session"] = session

    await session.start(
        agent=UnifiedTutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))