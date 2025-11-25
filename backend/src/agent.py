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

# Session state
session_state = {
    "current_mode": None,  # "learn", "quiz", or "teach_back"
    "current_concept": None,
    "concepts": []
}

def load_tutor_content() -> List[Dict[str, Any]]:
    """Load tutor content from JSON file"""
    if not os.path.exists(CONTENT_FILE):
        logger.error(f"Content file not found: {CONTENT_FILE}")
        return []
    
    try:
        with open(CONTENT_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        return []

def get_concept_by_id(concept_id: str) -> Dict[str, Any] | None:
    """Get a concept by ID"""
    concepts = session_state.get("concepts", [])
    for concept in concepts:
        if concept["id"] == concept_id:
            return concept
    return None

def get_available_concepts() -> str:
    """Get formatted list of available concepts"""
    concepts = session_state.get("concepts", [])
    if not concepts:
        return "No concepts available"
    
    concept_list = ", ".join([c["title"] for c in concepts])
    return f"Available topics: {concept_list}"


class CoordinatorAgent(Agent):
    """Main agent that greets and routes to learning modes"""
    
    def __init__(self) -> None:
        concepts_info = get_available_concepts()
        
        super().__init__(
            instructions=f"""You are the friendly coordinator for a Teach-the-Tutor learning system.

YOUR ROLE:
- Greet new users warmly
- Explain the three learning modes available
- Help users choose which mode to start with
- Switch between modes when requested

THREE LEARNING MODES:
1. LEARN mode - I explain concepts to you (perfect for beginners)
2. QUIZ mode - I ask you questions to test your knowledge
3. TEACH BACK mode - You explain concepts to me (best for mastery)

AVAILABLE CONTENT:
{concepts_info}

BEHAVIOR:
- When user first connects, greet them and ask which mode they'd like
- Use the switch_mode tool to transition between modes
- Keep your responses natural and encouraging
- One question or instruction at a time for voice

Example flow:
User connects → You greet → Explain modes → Ask preference → Use switch_mode tool

REMEMBER: You're just the coordinator. The actual learning happens in the specialist agents.""",
        )

    @function_tool
    async def switch_mode(
        self,
        context: RunContext,
        mode: str,
        concept_id: str = None
    ):
        """Switch to a different learning mode.
        
        Args:
            mode: The learning mode - must be "learn", "quiz", or "teach_back"
            concept_id: Optional concept ID to focus on (e.g., "variables", "loops")
        """
        mode = mode.lower()
        valid_modes = ["learn", "quiz", "teach_back"]
        
        if mode not in valid_modes:
            return f"Invalid mode. Please choose: {', '.join(valid_modes)}"
        
        session_state["current_mode"] = mode
        
        if concept_id:
            concept = get_concept_by_id(concept_id)
            if concept:
                session_state["current_concept"] = concept
                return f"Switching to {mode} mode for {concept['title']}. Transferring you now..."
            else:
                return f"Concept '{concept_id}' not found. Available: {get_available_concepts()}"
        
        return f"Switching to {mode} mode. Transferring you now..."

    @function_tool
    async def list_concepts(self, context: RunContext):
        """List all available concepts/topics"""
        return get_available_concepts()


class LearnModeAgent(Agent):
    """Learn mode - explains concepts (Matthew voice)"""
    
    def __init__(self) -> None:
        concepts_info = get_available_concepts()
        current_concept = session_state.get("current_concept")
        
        context = ""
        if current_concept:
            context = f"\n\nCURRENT TOPIC: {current_concept['title']}\nSUMMARY: {current_concept['summary']}"
        
        super().__init__(
            instructions=f"""You are Matthew, a friendly and clear teacher in LEARN mode.

YOUR ROLE:
- Explain concepts clearly and simply
- Use examples and analogies
- Break down complex ideas into digestible pieces
- Encourage questions

{concepts_info}{context}

TEACHING STYLE:
- Start with the big picture, then add details
- Use everyday examples
- Check for understanding
- Be patient and supportive
- Keep explanations concise for voice

If user wants to switch modes, acknowledge and tell them to ask the coordinator.

REMEMBER: You're Matthew, the explainer. Keep it clear and engaging!""",
        )

    @function_tool
    async def explain_concept(
        self,
        context: RunContext,
        concept_id: str
    ):
        """Explain a specific concept in detail.
        
        Args:
            concept_id: The ID of the concept to explain (e.g., "variables", "loops")
        """
        concept = get_concept_by_id(concept_id)
        if not concept:
            return f"I don't have information on that concept. {get_available_concepts()}"
        
        session_state["current_concept"] = concept
        
        return f"Let me explain {concept['title']}: {concept['summary']}"


class QuizModeAgent(Agent):
    """Quiz mode - asks questions (Alicia voice)"""
    
    def __init__(self) -> None:
        concepts_info = get_available_concepts()
        current_concept = session_state.get("current_concept")
        
        context = ""
        if current_concept:
            context = f"\n\nCURRENT TOPIC: {current_concept['title']}\nQUESTION: {current_concept['sample_question']}"
        
        super().__init__(
            instructions=f"""You are Alicia, an encouraging quiz master in QUIZ mode.

YOUR ROLE:
- Ask questions to test understanding
- Give helpful feedback on answers
- Encourage learning from mistakes
- Celebrate correct answers

{concepts_info}{context}

QUIZ STYLE:
- Ask one clear question at a time
- Wait for the full answer
- Provide constructive feedback
- If answer is incomplete, ask follow-up questions
- Keep tone supportive, never harsh

FEEDBACK GUIDELINES:
- Correct answers: Praise and maybe add a bonus insight
- Partially correct: Acknowledge what's right, gently guide to what's missing
- Incorrect: Be kind, explain the right answer, encourage trying again

If user wants to switch modes, acknowledge and tell them to ask the coordinator.

REMEMBER: You're Alicia, the quiz master. Make learning fun!""",
        )

    @function_tool
    async def ask_question(
        self,
        context: RunContext,
        concept_id: str
    ):
        """Ask a quiz question about a specific concept.
        
        Args:
            concept_id: The ID of the concept to quiz on
        """
        concept = get_concept_by_id(concept_id)
        if not concept:
            return f"I don't have a question for that. {get_available_concepts()}"
        
        session_state["current_concept"] = concept
        
        return f"Here's your question about {concept['title']}: {concept['sample_question']}"


class TeachBackModeAgent(Agent):
    """Teach back mode - listens and gives feedback (Ken voice)"""
    
    def __init__(self) -> None:
        concepts_info = get_available_concepts()
        current_concept = session_state.get("current_concept")
        
        context = ""
        if current_concept:
            context = f"\n\nCURRENT TOPIC: {current_concept['title']}\nKEY POINTS: {current_concept['summary']}"
        
        super().__init__(
            instructions=f"""You are Ken, a thoughtful listener and feedback provider in TEACH BACK mode.

YOUR ROLE:
- Ask users to teach concepts back to you
- Listen carefully to their explanations
- Provide constructive, specific feedback
- Highlight what they got right
- Gently correct misconceptions

{concepts_info}{context}

TEACH BACK APPROACH:
1. Invite them to explain the concept as if teaching you
2. Listen without interrupting
3. After they finish, give balanced feedback:
   - What they explained well
   - What they missed or got wrong
   - Overall assessment (Excellent/Good/Needs Work)

FEEDBACK STYLE:
- Start with positives
- Be specific about what was good
- Point out gaps gently
- End with encouragement
- Keep it conversational for voice

If user wants to switch modes, acknowledge and tell them to ask the coordinator.

REMEMBER: You're Ken, the listener. Help them learn through teaching!""",
        )

    @function_tool
    async def prompt_teach_back(
        self,
        context: RunContext,
        concept_id: str
    ):
        """Prompt user to teach back a concept.
        
        Args:
            concept_id: The ID of the concept for teach-back
        """
        concept = get_concept_by_id(concept_id)
        if not concept:
            return f"I don't have that concept. {get_available_concepts()}"
        
        session_state["current_concept"] = concept
        
        return f"Great! Now teach me about {concept['title']}. Explain it as if I know nothing about it."

    @function_tool
    async def give_feedback(
        self,
        context: RunContext,
        feedback: str,
        score: str = "Good"
    ):
        """Give feedback on the user's teach-back explanation.
        
        Args:
            feedback: Specific feedback on their explanation
            score: Overall assessment (Excellent/Good/Needs Work)
        """
        return f"Assessment: {score}\n\nFeedback: {feedback}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Load tutor content
    session_state["concepts"] = load_tutor_content()
    session_state["current_mode"] = "coordinator"
    
    logger.info(f"Loaded {len(session_state['concepts'])} concepts")

    # Determine which agent to use based on mode
    current_mode = session_state.get("current_mode", "coordinator")
    
    # Select voice and agent based on mode
    if current_mode == "learn":
        voice = "en-US-matthew"
        agent = LearnModeAgent()
    elif current_mode == "quiz":
        voice = "en-US-alicia"
        agent = QuizModeAgent()
    elif current_mode == "teach_back":
        voice = "en-US-ken"
        agent = TeachBackModeAgent()
    else:
        voice = "en-US-matthew"
        agent = CoordinatorAgent()

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

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))