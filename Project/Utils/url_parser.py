import re
from urllib.parse import urlparse, parse_qs, unquote
from typing import Dict, List, Optional

class URLParser:
    """Advanced URL parsing utilities"""
    
    @staticmethod
    def extract_all_segments(url: str) -> Dict[str, any]:
        """Extract all components from URL"""
        parsed = urlparse(url)
        
        return {
            'domain': parsed.netloc,
            'path': parsed.path,
            'path_segments': [s for s in parsed.path.split('/') if s],
            'query_params': parse_qs(parsed.query),
            'fragment': parsed.fragment,
            'scheme': parsed.scheme
        }
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for consistent processing"""
        # Remove trailing slashes
        url = url.rstrip('/')
        
        # Lowercase domain
        parsed = urlparse(url)
        normalized = parsed._replace(netloc=parsed.netloc.lower())
        
        return normalized.geturl()
    
    @staticmethod
    def extract_potential_slugs(url: str) -> List[str]:
        """Extract all potential slug candidates from URL"""
        parsed = urlparse(url)
        candidates = []
        
        # Path segments
        path_segments = [s for s in parsed.path.split('/') if s]
        candidates.extend(path_segments)
        
        # Query parameters that might contain slugs
        query_params = parse_qs(parsed.query)
        slug_param_names = ['product', 'item', 'p', 'id', 'sku', 'name']
        
        for param in slug_param_names:
            if param in query_params:
                candidates.extend(query_params[param])
        
        # Decode all candidates
        return [unquote(c) for c in candidates]