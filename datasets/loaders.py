import pandas as pd
from urllib.parse import urlparse
from pathlib import Path
from tqdm.auto import tqdm


class Loader:
    def __init__(self):
        pass
    
    
class WebTrackingLoader(Loader):
    def __init__(self, data_directory: str):
        self.user_data = self.load_user_data_from_csv(data_directory)

    @staticmethod
    def normalise_domain(url: str) -> str:
        """Extract and normalise domain from URL."""
        try:
            if not url.startswith("http"):
                url = f"http://{url}"

            # Parse URL and get netloc (domain part)
            domain = urlparse(url).netloc

            # Remove www. if present
            if domain.startswith("www."):
                domain = domain[4:]

            # Special handling for googlevideo.com domains
            if "googlevideo.com" in domain:
                return "googlevideo.com"

            # Split domain and get main domain + TLD
            parts = domain.split(".")

            # Handle subdomains
            if len(parts) > 2:
                # Special cases for known services where we want to preserve the subdomain
                special_cases = ["googleusercontent"]
                if any(case in domain for case in special_cases):
                    return domain

                # For other cases, take only the main domain and TLD
                return ".".join(parts[-2:])

            return domain
        except:
            return url

    @staticmethod
    def preprocess_user_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Load and preprocess the browsing data."""
        df["timestamp"] = pd.to_datetime(df["used_at"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["date"] = df["timestamp"].dt.date
        df["domain"] = df["domain"].apply(WebTrackingLoader.normalise_domain)
        cols = ["timestamp", "hour", "day_of_week", "date", "domain"]
        return df[cols].sort_values(by="timestamp")

    @staticmethod
    def load_user_data_from_csv(data_csv: str) -> dict[str, pd.DataFrame]:
        """Load and preprocess all user CSV files"""
        df = pd.read_csv(data_csv)
        user_data = {}

        users = df["panelist_id"].unique()
        print("Loading and preprocessing user data...")
        for user in tqdm(users):
            user_df = df[df["panelist_id"] == user].copy()
            user_data[user] = WebTrackingLoader.preprocess_user_frame(user_df)

        print(f"Loaded data for {len(user_data)} users")
        return user_data    


class AliceLoader(Loader):
    def __init__(self, data_directory: str):
        self.user_data = self.load_user_data_from_directory(data_directory)

    @staticmethod
    def normalise_domain(url: str) -> str:
        """Extract and normalise domain from URL."""
        try:
            if not url.startswith("http"):
                url = f"http://{url}"

            # Parse URL and get netloc (domain part)
            domain = urlparse(url).netloc

            # Remove www. if present
            if domain.startswith("www."):
                domain = domain[4:]

            # Special handling for googlevideo.com domains
            if "googlevideo.com" in domain:
                return "googlevideo.com"

            # Split domain and get main domain + TLD
            parts = domain.split(".")

            # Handle subdomains
            if len(parts) > 2:
                # Special cases for known services where we want to preserve the subdomain
                special_cases = ["googleusercontent"]
                if any(case in domain for case in special_cases):
                    return domain

                # For other cases, take only the main domain and TLD
                return ".".join(parts[-2:])

            return domain
        except:
            return url

    @staticmethod
    def preprocess_user_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Load and preprocess the browsing data."""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["date"] = df["timestamp"].dt.date
        df["domain"] = df["site"].apply(AliceLoader.normalise_domain)
        return df

    @staticmethod
    def load_user_data_from_directory(data_directory: str) -> dict[str, pd.DataFrame]:
        """Load and preprocess all user CSV files"""
        data_dir = Path(data_directory)
        user_data = {}

        print("Loading and preprocessing user data...")
        for file_path in data_dir.glob("*.csv"):
            user_id = file_path.stem
            df = pd.read_csv(file_path)
            user_data[user_id] = AliceLoader.preprocess_user_frame(df)

        print(f"Loaded data for {len(user_data)} users")
        return user_data
