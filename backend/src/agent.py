import logging
import json
import os
from datetime import datetime

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

# Order state - tracks the current coffee order
order_state = {
    "drinkType": None,
    "size": None,
    "milk": None,
    "extras": [],
    "name": None
}

def reset_order():
    """Reset order state for a new order"""
    global order_state
    order_state = {
        "drinkType": None,
        "size": None,
        "milk": None,
        "extras": [],
        "name": None
    }

def get_missing_fields():
    """Get list of fields that still need to be filled"""
    missing = []
    if not order_state["drinkType"]:
        missing.append("drink type")
    if not order_state["size"]:
        missing.append("size")
    if not order_state["milk"]:
        missing.append("milk preference")
    if not order_state["name"]:
        missing.append("name for the order")
    return missing


class CoffeeBarista(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and enthusiastic coffee shop barista at "Brew Haven Cafe". 
            
Your job is to take coffee orders from customers through voice conversation.

MENU OPTIONS:
- Drink Types: Espresso, Americano, Latte, Cappuccino, Mocha, Macchiato, Cold Brew, Iced Coffee, Hot Chocolate, Chai Latte
- Sizes: Small, Medium, Large
- Milk Options: Regular, Oat, Almond, Soy, Coconut, Skim, None (for black coffee)
- Extras: Whipped Cream, Extra Shot, Vanilla Syrup, Caramel Syrup, Hazelnut Syrup, Chocolate Drizzle, Cinnamon

BEHAVIOR:
1. Greet customers warmly when they arrive
2. Ask what drink they would like to order
3. Ask clarifying questions ONE AT A TIME until you have: drink type, size, milk preference, and customer name
4. For extras, ask if they want to add anything extra (whipped cream, syrups, etc.) - this is optional
5. Once you have all required information, use the update_order tool to save each field
6. When the order is complete, use the save_order tool to finalize it
7. Confirm the complete order back to the customer

IMPORTANT:
- Be conversational and friendly, not robotic
- Ask only ONE question at a time
- Use the tools to update and save the order
- Keep responses concise and natural for voice
- Don't use emojis, asterisks, or special formatting
- If customer says "no extras" or similar, that's fine - extras are optional""",
        )

    @function_tool
    async def update_order(
        self,
        context: RunContext,
        field: str,
        value: str
    ):
        """Update a field in the current coffee order.
        
        Args:
            field: The field to update. Must be one of: drinkType, size, milk, extras, name
            value: The value to set for the field. For extras, this adds to the list.
        """
        global order_state
        
        field = field.lower()
        field_map = {
            "drinktype": "drinkType",
            "drink_type": "drinkType",
            "drink": "drinkType",
            "size": "size",
            "milk": "milk",
            "extras": "extras",
            "extra": "extras",
            "name": "name",
            "customer_name": "name"
        }
        
        actual_field = field_map.get(field, field)
        
        if actual_field == "extras":
            if value.lower() not in ["none", "no", "nothing"]:
                order_state["extras"].append(value)
            logger.info(f"Added extra: {value}")
        elif actual_field in order_state:
            order_state[actual_field] = value
            logger.info(f"Updated {actual_field}: {value}")
        else:
            return f"Unknown field: {field}"
        
        missing = get_missing_fields()
        if missing:
            return f"Order updated. Still need: {', '.join(missing)}"
        else:
            return "Order complete! All required fields are filled. Please use save_order to finalize."

    @function_tool
    async def save_order(self, context: RunContext):
        """Save the completed order to a JSON file. Call this when all required fields are filled."""
        global order_state
        
        missing = get_missing_fields()
        if missing:
            return f"Cannot save order yet. Still need: {', '.join(missing)}"
        
        # Create orders directory if it doesn't exist
        os.makedirs("orders", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orders/order_{timestamp}.json"
        
        # Prepare order data
        order_data = {
            "order": order_state.copy(),
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Save to JSON file
        with open(filename, "w") as f:
            json.dump(order_data, f, indent=2)
        
        logger.info(f"Order saved to {filename}")
        
        # Create order summary
        extras_text = ", ".join(order_state["extras"]) if order_state["extras"] else "none"
        summary = f"""Order saved successfully!
        
Order for: {order_state['name']}
Drink: {order_state['size']} {order_state['drinkType']}
Milk: {order_state['milk']}
Extras: {extras_text}

Order saved to {filename}"""
        
        # Reset for next order
        reset_order()
        
        return summary

    @function_tool
    async def get_order_status(self, context: RunContext):
        """Get the current status of the order being taken."""
        missing = get_missing_fields()
        extras_text = ", ".join(order_state["extras"]) if order_state["extras"] else "none yet"
        
        status = f"""Current order status:
Drink: {order_state['drinkType'] or 'not set'}
Size: {order_state['size'] or 'not set'}
Milk: {order_state['milk'] or 'not set'}
Extras: {extras_text}
Name: {order_state['name'] or 'not set'}

Missing fields: {', '.join(missing) if missing else 'None - order ready to save!'}"""
        
        return status


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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
        agent=CoffeeBarista(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))