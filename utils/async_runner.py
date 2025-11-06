import asyncio

def run_async(coro):
    """Run async coroutine in sync context"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)