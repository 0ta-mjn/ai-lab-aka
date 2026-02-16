from urllib.parse import urlparse


def is_same_domain(url: str, company_url: str) -> bool:
    """Check if the URL belongs to the same domain as the company URL."""
    try:
        parsed_url = urlparse(url)
        parsed_company = urlparse(company_url)

        if parsed_url.scheme not in ["http", "https"]:
            return False

        # Simple domain matching (ignoring www.)
        domain_url = parsed_url.netloc.lower().replace("www.", "")
        domain_company = parsed_company.netloc.lower().replace("www.", "")

        return domain_url == domain_company
    except Exception:
        return False
