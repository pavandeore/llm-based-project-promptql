from sqlalchemy import create_engine, MetaData
from services.database.schema_manager import DatabaseSchemaManager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and schema extraction"""
    
    def __init__(self):
        self.schema_manager = DatabaseSchemaManager()
    
    def extract_detailed_schema_with_relationships(self, engine, db_type: str):
        """Extract schema with foreign keys and relationships"""
        schema_data = []
        metadata = MetaData()
        
        try:
            with engine.connect() as conn:
                # Reflect all tables with foreign keys
                metadata.reflect(bind=engine)
                
                for table_name, table in metadata.tables.items():
                    table_info = {
                        "table_name": table_name,
                        "columns": [],
                        "column_types": [],
                        "descriptions": [],
                        "primary_keys": [col.name for col in table.primary_key],
                        "foreign_keys": [],
                        "relationships": [],
                        "database_type": db_type
                    }
                    
                    # Extract columns and foreign keys
                    for column in table.columns:
                        table_info["columns"].append(column.name)
                        table_info["column_types"].append(str(column.type))
                        
                        # Extract foreign key relationships
                        for fk in column.foreign_keys:
                            fk_info = {
                                "column": column.name,
                                "references_table": fk.column.table.name,
                                "references_column": fk.column.name
                            }
                            table_info["foreign_keys"].append(fk_info)
                            table_info["relationships"].append(
                                f"REFERENCES {fk.column.table.name}({fk.column.name})"
                            )
                    
                    schema_data.append(table_info)
            
            return schema_data
            
        except Exception as e:
            logger.error(f"Error extracting schema with relationships: {e}")
            raise

    async def extract_and_store_schema_async(self, engine, db_type: str):
        """Extract schema and store embeddings asynchronously"""
        schema_data = self.extract_detailed_schema_with_relationships(engine, db_type)
        await self.schema_manager.store_schema_embeddings_async(schema_data, db_type)
        return schema_data

    async def get_enhanced_schema_for_query_async(self, query: str, n_results: int = 5) -> str:
        """Get relevant schema with relationship context asynchronously"""
        relevant_tables = await self.schema_manager.get_relevant_schema_async(query, n_results)
        
        schema_info = "Database Schema with Relationships:\n\n"
        
        for table in relevant_tables:
            schema_info += f"📊 Table: {table['table_name']}\n"
            schema_info += f"   Columns: {', '.join(table['columns'])}\n"
            
            # Add primary keys
            if table.get('primary_keys'):
                schema_info += f"   Primary Keys: {', '.join(table['primary_keys'])}\n"
            
            # Add foreign keys and relationships
            if table.get('foreign_keys'):
                schema_info += "   Relationships:\n"
                for fk in table['foreign_keys']:
                    schema_info += f"     - {fk['column']} → {fk['references_table']}.{fk['references_column']}\n"
            
            schema_info += "\n"
        
        return schema_info