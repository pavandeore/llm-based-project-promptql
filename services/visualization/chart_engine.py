from services.llm.visualization_llm import analyze_visualization_intent, create_detailed_chart_config, determine_smart_chart_type
import logging

logger = logging.getLogger(__name__)

async def determine_smart_chart_type_multi_step(rows, columns, user_query):
    """
    Multi-step chart determination process for higher quality visualizations.
    """
    if not rows or not columns:
        return {
            "chart_type": "table",
            "description": "No data available for visualization",
            "series_config": [],
            "recommended_columns": columns if columns else [],
            "all_charts": []
        }

    try:
        # Step 1: Analyze intent and get chart suggestions
        logger.info("🔍 Step 1: Analyzing visualization intent...")
        intent_analysis = await analyze_visualization_intent(rows, columns, user_query)
        
        if not intent_analysis.get("suggestions"):
            logger.warning("No chart suggestions generated, falling back to table")
            return {
                "chart_type": "table",
                "description": intent_analysis.get("primary_insight", "No specific visualizations suggested"),
                "series_config": [],
                "recommended_columns": columns[:4] if columns else [],
                "all_charts": []
            }

        # Step 2: Create detailed config for the primary chart (first suggestion)
        logger.info("🎨 Step 2: Creating detailed chart configuration...")
        primary_suggestion = intent_analysis["suggestions"][0]
        detailed_config = await create_detailed_chart_config(
            primary_suggestion, rows, columns, user_query
        )

        if not detailed_config:
            logger.warning("Detailed config failed, using fallback")
            return {
                "chart_type": primary_suggestion.get("chart_type", "table"),
                "description": primary_suggestion.get("purpose", ""),
                "series_config": [],
                "recommended_columns": primary_suggestion.get("required_columns", columns[:4] if columns else []),
                "all_charts": intent_analysis["suggestions"]
            }

        # Return comprehensive result
        return {
            "chart_type": detailed_config.get("chart_type", "table"),
            "chart_title": detailed_config.get("chart_title", ""),
            "description": detailed_config.get("chart_description", primary_suggestion.get("purpose", "")),
            "series_config": detailed_config.get("series_config", []),
            "recommended_columns": detailed_config.get("recommended_columns", columns[:4] if columns else []),
            "all_charts": intent_analysis["suggestions"],
            "primary_insight": intent_analysis.get("primary_insight", ""),
            "x_axis": detailed_config.get("x_axis", {}),
            "y_axis": detailed_config.get("y_axis", {}),
            "data_transformations": detailed_config.get("data_transformations", [])
        }

    except Exception as e:
        logger.error(f"Multi-step chart determination failed: {e}")
        # Fallback to original method with proper error handling
        try:
            return await determine_smart_chart_type(rows, columns, user_query)
        except Exception as fallback_error:
            logger.error(f"Fallback chart determination also failed: {fallback_error}")
            return {
                "chart_type": "table",
                "description": f"Visualization failed: {str(e)}",
                "series_config": [],
                "recommended_columns": columns[:4] if columns else [],
                "all_charts": []
            }