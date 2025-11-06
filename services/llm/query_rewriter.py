from extensions import aclient
import logging

logger = logging.getLogger(__name__)

async def rewrite_user_query_for_quality(user_query: str) -> str:
    """
    Uses GPT to rewrite a natural language query into a clearer, more structured analytical question.
    """
    try:
        prompt = f"""
You are an expert data analyst and business intelligence query rewriter.

Your task is to rewrite the following natural language question
into a clearer, more analytical form suitable for generating SQL queries and visualizations.

Keep the same meaning, but:
- Clarify metrics and dimensions
- Add context like averages, totals, comparisons, trends, relationships, or time periods
- Use business-friendly but structured phrasing
- Prefer questions that can be visualized (scatter, trend, comparison, distribution)

Example transformations:
- "sales by region" → "Compare total sales across different regions"
- "trend of signups" → "Show monthly trend of user signups over time"
- "relationship between price and rating" → "Compare average rating against course price for all visible courses"

USER QUERY:
{user_query}

Return ONLY the rewritten question.
"""

        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You rewrite vague business questions into clear, analytical ones."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100,
        )

        rewritten = response.choices[0].message.content.strip()
        logger.info(f"🪄 Rewritten Query: {rewritten}")
        return rewritten

    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return user_query  # fallback to original