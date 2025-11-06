def get_db_url(db_type: str, db_host: str, db_port: str, db_name: str, db_user: str, db_password: str) -> str:
    """Generate the appropriate database URL based on the database type"""
    if db_type == 'postgresql':
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == 'mysql':
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == 'sqlite':
        return f"sqlite:///{db_name}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")