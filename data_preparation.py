# src/data_preparation.py
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

def generate_support_tickets(num_samples=250):
    """
    Generate realistic support ticket data for training
    """
    
    # Billing-related tickets
    billing_templates = [
        "I was charged twice for my subscription this month",
        "My payment failed but I still have access, please clarify billing",
        "Can you help me understand the charges on my last invoice?",
        "I need to update my payment method urgently",
        "My credit card was declined, how do I fix billing issues?",
        "Invoice #{} shows incorrect amount, please review",
        "Automatic billing charged wrong account, need refund",
        "Subscription cancelled but still being charged monthly",
        "Pro plan billing started early, please adjust charges",
        "Payment processing failed multiple times this week"
    ]
    
    # Technical-related tickets  
    technical_templates = [
        "Application crashes when I try to upload large files",
        "API returning 500 errors consistently since yesterday",
        "Login page not loading, getting timeout errors",
        "Dashboard widgets not displaying data correctly",
        "Integration with {} service stopped working",
        "Performance is very slow during peak hours",
        "Mobile app crashes on iOS 17 devices",
        "Database connection errors in production environment",
        "SSL certificate expired, getting security warnings", 
        "Search functionality returning empty results"
    ]
    
    # Other category tickets
    other_templates = [
        "How do I add new team members to my workspace?",
        "Can you provide training materials for new users?",
        "What are the data retention policies for my account?",
        "I need help setting up single sign-on integration",
        "Account security best practices documentation needed",
        "How to export my data before subscription ends?",
        "Feature request: dark mode for the dashboard",
        "Partnership opportunities with your company",
        "Can I get a demo of premium features?",
        "Compliance requirements for healthcare industry use"
    ]
    
    tickets = []
    labels = []
    
    # Generate tickets for each category
    categories = {
        'Billing': billing_templates,
        'Technical': technical_templates, 
        'Other': other_templates
    }
    
    samples_per_category = num_samples // 3
    
    for category, templates in categories.items():
        for i in range(samples_per_category):
            # Add variety with additional context
            template = random.choice(templates)
            
            # Add realistic variations
            if '{}' in template:
                services = ['Slack', 'Salesforce', 'GitHub', 'Jira', 'Google Drive']
                template = template.format(random.choice(services))
            
            # Add context variations
            contexts = [
                template,
                f"Urgent: {template}",
                f"Follow-up: {template}",
                f"{template} Please help ASAP.",
                f"{template} This is affecting our entire team.",
            ]
            
            ticket_text = random.choice(contexts)
            tickets.append(ticket_text)
            labels.append(category)
    
    return pd.DataFrame({
        'text': tickets,
        'label': labels
    })

if __name__ == "__main__":
    # Generate dataset
    df = generate_support_tickets(500)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('data/support_tickets.csv', index=False)
    
    print(f"Generated {len(df)} support tickets")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nSample tickets:")
    for label in df['label'].unique():
        print(f"\n{label} example:")
        sample = df[df['label'] == label].iloc[0]['text']
        print(f"  {sample}")
