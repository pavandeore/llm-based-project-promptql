import pandas as pd
import json
from extensions import aclient
import logging

logger = logging.getLogger(__name__)

async def analyze_visualization_intent(rows, columns, user_query):
    """
    Step 1: Analyze user intent and suggest appropriate chart types with specific column mappings.
    Returns a list of chart suggestions with their purpose and required columns.
    """
    if not rows or not columns:
        return {"suggestions": [], "primary_insight": "No data available", "all_columns": columns}

    try:
        # Convert rows to proper format for DataFrame
        # Handle both SQLAlchemy Row objects and regular tuples/dicts
        if hasattr(rows[0], '_asdict'):
            # SQLAlchemy Row objects
            row_dicts = [row._asdict() for row in rows]
        elif isinstance(rows[0], (tuple, list)):
            # Regular tuples/lists
            row_dicts = [dict(zip(columns, row)) for row in rows]
        else:
            # Assume already dictionaries
            row_dicts = rows
        
        df = pd.DataFrame(row_dicts)
        
        # Ensure all columns are strings
        df.columns = [str(col) for col in df.columns]
        
        df_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict()
        }

        prompt = f"""
You are a senior data analyst. Analyze the user's query and dataset to suggest the most meaningful visualizations.

USER QUERY: {user_query}

DATASET OVERVIEW:
- Shape: {df_info['shape']}
- Columns: {df_info['columns']}
- Data Types: {df_info['dtypes']}
- Unique Values per Column: {df_info['unique_counts']}

ANALYSIS TASK:
1. Understand the user's analytical intent from their query
2. Suggest 1-3 specific chart types that would best answer their question
3. For each chart, specify EXACT which columns to use and why

CHART TYPE CATEGORIES:
- COMPARISON: bar, column, grouped_bar (comparing categories)
- TREND: line, area (showing changes over time/sequence)
- DISTRIBUTION: histogram, box_plot, scatter (showing data spread)
- COMPOSITION: pie, stacked_bar, donut (showing parts of whole)
- RELATIONSHIP: scatter, bubble (showing correlations)
- METRIC: big_number, gauge (showing key metrics)

OUTPUT FORMAT (JSON):
{{
  "chart_suggestions": [
    {{
      "chart_name": "Descriptive name for this visualization",
      "chart_type": "bar|line|pie|scatter|combo|histogram|big_number|etc",
      "purpose": "What business question this chart answers",
      "required_columns": ["col1", "col2", "col3"],  // MAX 4 columns
      "primary_metric": "column_name",  // main metric being measured
      "breakdown_by": "column_name",    // dimension for breakdown
      "time_series": "column_name"      // if time-based analysis
    }}
  ],
  "primary_insight": "Overall main insight from the data based on user query"
}}

EXAMPLES:
For "Show sales by region":
{{
  "chart_suggestions": [
    {{
      "chart_name": "Sales by Region",
      "chart_type": "bar",
      "purpose": "Compare total sales across different regions",
      "required_columns": ["region", "sales_amount"],
      "primary_metric": "sales_amount",
      "breakdown_by": "region"
    }}
  ]
}}

For "Revenue trend over time":
{{
  "chart_suggestions": [
    {{
      "chart_name": "Monthly Revenue Trend",
      "chart_type": "line", 
      "purpose": "Show revenue growth/decline over time",
      "required_columns": ["month", "revenue"],
      "primary_metric": "revenue",
      "time_series": "month"
    }}
  ]
}}
"""

        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data visualization strategist that understands business context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        
        # Validate and limit to max 3 suggestions
        suggestions = result.get("chart_suggestions", [])[:3]
        
        # Ensure each suggestion has max 4 columns and columns exist in dataframe
        valid_columns = set(df.columns)
        for suggestion in suggestions:
            if "required_columns" in suggestion:
                # Filter to only include columns that actually exist
                suggestion["required_columns"] = [
                    col for col in suggestion["required_columns"][:4] 
                    if col in valid_columns
                ]
        
        return {
            "suggestions": suggestions,
            "primary_insight": result.get("primary_insight", ""),
            "all_columns": list(df.columns)
        }

    except Exception as e:
        logger.error(f"Visualization intent analysis failed: {e}")
        return {"suggestions": [], "primary_insight": f"Analysis failed: {str(e)}", "all_columns": columns}

async def create_detailed_chart_config(chart_suggestion, rows, columns, user_query):
    """
    Step 2: Create detailed configuration for a specific chart suggestion.
    """
    if not chart_suggestion or not rows:
        return None

    try:
        # Convert rows to proper format for DataFrame
        if hasattr(rows[0], '_asdict'):
            row_dicts = [row._asdict() for row in rows]
        elif isinstance(rows[0], (tuple, list)):
            row_dicts = [dict(zip(columns, row)) for row in rows]
        else:
            row_dicts = rows
        
        df = pd.DataFrame(row_dicts)
        df.columns = [str(col) for col in df.columns]
        
        required_cols = chart_suggestion.get("required_columns", [])
        
        # Filter dataframe to only required columns that exist
        available_cols = [col for col in required_cols if col in df.columns]
        if not available_cols:
            available_cols = df.columns.tolist()[:4]  # fallback to first 4 columns
            
        chart_df = df[available_cols].copy()
        
        # Sample data for context
        sample_data = chart_df.head(10).to_dict('records')
        
        prompt = f"""
You are a chart configuration expert. Create detailed visualization specifications.

CHART REQUEST:
- User Query: {user_query}
- Chart Name: {chart_suggestion.get('chart_name', 'Unknown')}
- Chart Type: {chart_suggestion.get('chart_type', 'bar')}
- Purpose: {chart_suggestion.get('purpose', '')}
- Required Columns: {available_cols}

SAMPLE DATA (first 10 rows):
{json.dumps(sample_data, indent=2, default=str)}

CHART CONFIGURATION TASK:
Create a complete chart configuration including:
1. Exact series mapping (x-axis, y-axis, categories)
2. Appropriate aggregations (sum, count, average)
3. Color schemes and styling recommendations
4. Axis labels and formatting
5. Any data transformations needed

OUTPUT FORMAT (JSON):
{{
  "chart_type": "{chart_suggestion.get('chart_type', 'bar')}",
  "chart_title": "Clear, descriptive title",
  "chart_description": "What this chart shows",
  "recommended_columns": {available_cols},
  "series_config": [
    {{
      "type": "bar|line|area|pie|scatter",
      "x": "column_for_x_axis",
      "y": "column_for_y_axis", 
      "aggregation": "sum|count|avg|min|max|none",
      "name": "Series display name",
      "color": "#hex_color",  // optional
      "stacking": true|false  // for bar charts
    }}
  ],
  "x_axis": {{
    "title": "X Axis Label",
    "type": "category|datetime|numeric"
  }},
  "y_axis": {{
    "title": "Y Axis Label", 
    "type": "numeric"
  }},
  "data_transformations": [
    "Sort by x-axis ascending",
    "Filter out null values in y-axis"
  ]
}}

EXAMPLES:

For bar chart comparing categories:
{{
  "chart_type": "bar",
  "chart_title": "Sales by Region",
  "series_config": [
    {{
      "type": "bar",
      "x": "region",
      "y": "sales_amount", 
      "aggregation": "sum",
      "name": "Total Sales"
    }}
  ]
}}

For time series line chart:
{{
  "chart_type": "line", 
  "chart_title": "Revenue Trend Over Time",
  "series_config": [
    {{
      "type": "line",
      "x": "month",
      "y": "revenue",
      "aggregation": "sum", 
      "name": "Monthly Revenue"
    }}
  ]
}}
"""

        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You create precise chart configurations for data visualization libraries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        config = json.loads(response.choices[0].message.content)
        
        # Add original suggestion context
        config["original_suggestion"] = chart_suggestion
        
        return config

    except Exception as e:
        logger.error(f"Detailed chart config failed: {e}")
        return None

async def determine_smart_chart_type(rows, columns, user_query):
    """
    Uses LLM to pick meaningful charts and select only 2-4 relevant columns.
    Uses pandas for robust type detection and data analysis.
    """
    if not rows or not columns:
        return {
            "chart_type": "table",
            "description": "No data available for visualization",
            "series_config": [],
            "recommended_columns": columns if columns else []
        }

    try:
        # Convert rows to proper format
        if hasattr(rows[0], '_asdict'):
            row_dicts = [row._asdict() for row in rows]
        elif isinstance(rows[0], (tuple, list)):
            row_dicts = [dict(zip(columns, row)) for row in rows]
        else:
            row_dicts = rows
        
        df = pd.DataFrame(row_dicts)
        df.columns = [str(col) for col in df.columns]
        
        # Basic DataFrame info for the prompt
        df_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict()
        }

        # Enhanced prompt with data insights
        prompt = f"""
You are an expert data visualization analyst. Your task is to:

1. Analyze the dataset and select ONLY 2-4 most meaningful columns for visualization
2. Recommend the best chart type(s) - can be simple (bar, line, pie) or combined (line+bar)
3. Provide clear series configuration for the visualization

User Query: {user_query}

DATASET OVERVIEW:
- Shape: {df_info['shape']}
- Columns: {df_info['columns']}
- Data Types: {df_info['dtypes']}

SELECTION CRITERIA:
- Prefer columns with meaningful patterns (categories, trends, distributions)
- Avoid ID columns, timestamps (unless for time series), or high-cardinality text
- For large datasets (>3 cols), focus on the most insightful 2-4 columns
- Consider relationships between columns (x-axis vs y-axis)

CHART TYPE GUIDELINES:
- Bar charts: Compare categories with numeric values
- Line charts: Show trends over time/sequence
- Pie charts: Show proportions (use sparingly, max 6-8 categories)
- Scatter plots: Show relationships between two numeric variables
- Combo charts: Use when multiple series need different visual encodings

RETURN STRICT JSON FORMAT:
{{
"chart_type": "bar" | "line" | "pie" | "scatter" | "combo" | "table",
"description": "Brief reasoning for chart and column selection",
"recommended_columns": ["col1", "col2", ...],  // MAX 4 COLUMNS
"series_config": [
    {{
    "type": "bar" | "line" | "area" | "pie",
    "x": "column_for_x_axis",
    "y": "column_for_y_axis",  // optional for pie charts
    "aggregation": "sum" | "count" | "avg" | null,
    "name": "Series display name"
    }}
]
}}

IMPORTANT: If the data is too complex or has too many text columns, recommend "table" visualization.
"""

        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior data visualization expert that selects only the most meaningful columns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        
        # Validate and ensure recommended_columns has max 4 columns
        if "recommended_columns" in result and len(result["recommended_columns"]) > 4:
            result["recommended_columns"] = result["recommended_columns"][:4]
            result["description"] += " (Limited to 4 most relevant columns)"
            
        return result

    except Exception as e:
        logger.error(f"Chart type decision failed: {e}")
        # Fallback: use original columns (limited to 4)
        fallback_columns = columns[:4] if columns and len(columns) > 4 else (columns if columns else [])
        return {
            "chart_type": "table",
            "description": f"Fallback due to error: {str(e)}",
            "series_config": [],
            "recommended_columns": fallback_columns
        }