#!/usr/bin/env python3
"""
Day 2 Challenge Solution: Website Summarizer using Ollama

This module upgrades the Day 1 project to use an Open Source model
running locally via Ollama rather than OpenAI.

Benefits:
1. No API charges - open-source
2. Data doesn't leave your box

Disadvantages:
1. Significantly less power than Frontier Model

Usage:
    Command line:
        uv run week1/solution.py <url>
    
    In Jupyter notebook:
        from solution import display_summary_website
        display_summary_website("https://edwarddonner.com")
    
Example:
    uv run week1/solution.py https://edwarddonner.com
"""

import sys
from openai import OpenAI
from scraper import fetch_website_contents
from IPython.display import Markdown, display


# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"  # Use "llama3.2:1b" if Ollama is slow on your machine


def create_ollama_client():
    """Create and return an OpenAI-compatible client for Ollama."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')


def create_messages(website_content):
    """
    Create the messages list for the LLM.
    
    Args:
        website_content: The scraped content from the website
        
    Returns:
        List of message dictionaries with system and user prompts
    """
    system_prompt = """
You are a helpful assistant that analyzes the contents of a website,
and provides a short, informative summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""
    
    user_prompt = f"""
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

{website_content}
"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def summarize_website(url, client=None):
    """
    Fetch and summarize a website using Ollama.
    
    Args:
        url: The URL of the website to summarize
        client: Optional OpenAI client (creates one if not provided)
        
    Returns:
        String containing the summary in markdown format
    """
    if client is None:
        client = create_ollama_client()
    
    # Fetch website content
    website_content = fetch_website_contents(url)
    
    # Create messages
    messages = create_messages(website_content)
    
    # Call Ollama
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    
    return response.choices[0].message.content


def display_summary_website(url):
    """
    Display a formatted summary of a website using IPython's Markdown display.
    This function is designed for use in Jupyter notebooks.
    
    Args:
        url: The URL of the website to summarize
    """
    summary = summarize_website(url)
    display(Markdown(summary))


def main():
    """Main function to run the website summarizer from command line."""
    if len(sys.argv) < 2:
        print("Usage: uv run week1/solution.py <url>")
        print("Example: uv run week1/solution.py https://edwarddonner.com")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Validate URL format
    if not url.startswith(('http://', 'https://')):
        print("Error: URL must start with http:// or https://")
        sys.exit(1)
    
    try:
        # Get the summary
        summary = summarize_website(url)
        
        # Display in console with formatting
        print("\n" + "="*80)
        print("WEBSITE SUMMARY")
        print("="*80 + "\n")
        print(summary)
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is installed and running (visit http://localhost:11434/)")
        print("2. Make sure you've pulled the model: ollama pull llama3.2")
        print("3. If Ollama is slow, try using llama3.2:1b instead")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
