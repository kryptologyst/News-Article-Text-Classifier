"""
Mock News Database Generator
Generates a larger, more realistic dataset for news classification
"""

import pandas as pd
import random
from datetime import datetime, timedelta

class NewsDataGenerator:
    def __init__(self):
        self.categories = ['Politics', 'Sports', 'Technology', 'Finance', 'Science', 'Health', 'Entertainment']
        
        # Template headlines for each category
        self.templates = {
            'Politics': [
                'Government announces new {policy} reforms',
                'Elections bring {outcome} in {location}',
                'Political leaders debate {topic} strategies',
                'New legislation on {subject} passes parliament',
                'Prime Minister discusses {issue} with cabinet',
                'Opposition party criticizes {policy} decisions',
                'Voters express concerns about {topic}',
                'International relations improve with {country}',
                'Political scandal emerges regarding {subject}',
                'New ambassador appointed to {country}'
            ],
            'Sports': [
                '{team} wins {tournament} championship',
                '{sport} season ends with dramatic {event}',
                'Olympic athlete breaks {record} record',
                'Local {sport} club advances to finals',
                '{player} signs contract with {team}',
                'Stadium renovation completed for {sport}',
                'Coach announces retirement after {years} years',
                'Injury sidelines star {position} for season',
                'New sports facility opens in {location}',
                'Youth {sport} program launches nationwide'
            ],
            'Technology': [
                'Tech companies unveil latest {technology} breakthroughs',
                'AI revolution transforms {industry} sector',
                'New smartphone features {innovation}',
                'Cybersecurity threats target {sector}',
                'Startup develops innovative {solution}',
                'Tech giants invest in {technology} research',
                'Software update improves {feature}',
                'Data privacy concerns rise over {platform}',
                'Quantum computing advances in {application}',
                'Tech conference showcases {innovation}'
            ],
            'Finance': [
                'Stock markets {direction} after {news}',
                'Cryptocurrency {action} following {event}',
                'Banking sector reports {outcome}',
                'Interest rates {change} by central bank',
                'Economic growth shows {trend} signs',
                'Investment firm launches new {product}',
                'Trade negotiations affect {market}',
                'Inflation concerns impact {sector}',
                'Financial regulations updated for {industry}',
                'Market volatility increases due to {factor}'
            ],
            'Science': [
                'Scientists discover new {discovery} in {field}',
                'Research breakthrough in {area} announced',
                'Climate study reveals {finding}',
                'Space mission explores {destination}',
                'Medical research advances {treatment}',
                'Environmental study shows {result}',
                'Archaeological team uncovers {artifact}',
                'Laboratory develops new {technology}',
                'Scientific collaboration yields {outcome}',
                'Research funding approved for {project}'
            ],
            'Health': [
                'New treatment shows promise for {condition}',
                'Health officials report {statistic}',
                'Medical breakthrough in {field} research',
                'Vaccine development progresses for {disease}',
                'Public health campaign targets {issue}',
                'Hospital introduces new {technology}',
                'Study links {factor} to {condition}',
                'Mental health awareness increases in {demographic}',
                'Fitness trends promote {activity}',
                'Nutrition research reveals {finding}'
            ],
            'Entertainment': [
                'Movie {title} breaks box office records',
                'Celebrity {name} announces new {project}',
                'Music festival features {genre} artists',
                'TV series {title} wins {award}',
                'Theater production of {play} opens',
                'Streaming platform launches {content}',
                'Gaming industry reports {statistic}',
                'Art exhibition showcases {style}',
                'Book {title} becomes bestseller',
                'Entertainment venue opens in {location}'
            ]
        }
        
        # Filler words for templates
        self.fillers = {
            'policy': ['healthcare', 'education', 'tax', 'immigration', 'environmental', 'economic'],
            'outcome': ['significant changes', 'unexpected results', 'historic victories', 'close contests'],
            'location': ['the capital', 'major cities', 'rural areas', 'the region', 'key states'],
            'topic': ['economic', 'social', 'environmental', 'foreign policy', 'domestic'],
            'subject': ['budget', 'infrastructure', 'security', 'trade', 'welfare'],
            'issue': ['climate change', 'economic recovery', 'social reforms', 'international trade'],
            'country': ['neighboring nations', 'allied countries', 'trade partners', 'regional powers'],
            'team': ['Manchester United', 'Lakers', 'Patriots', 'Warriors', 'City FC'],
            'tournament': ['Champions League', 'World Cup', 'NBA Finals', 'Super Bowl'],
            'sport': ['football', 'basketball', 'tennis', 'soccer', 'baseball'],
            'event': ['finale', 'overtime', 'penalty shootout', 'comeback'],
            'player': ['Star player', 'Veteran athlete', 'Rising talent', 'Team captain'],
            'record': ['world', 'national', 'Olympic', 'season'],
            'position': ['quarterback', 'striker', 'point guard', 'pitcher'],
            'years': ['10', '15', '20', '25'],
            'technology': ['AI', 'blockchain', 'quantum computing', '5G', 'IoT'],
            'industry': ['healthcare', 'finance', 'manufacturing', 'retail', 'education'],
            'innovation': ['advanced camera', 'longer battery', 'faster processor', 'new design'],
            'sector': ['banking', 'government', 'healthcare', 'education', 'retail'],
            'solution': ['app', 'platform', 'device', 'system', 'tool'],
            'platform': ['social media', 'messaging apps', 'cloud services', 'e-commerce'],
            'application': ['medicine', 'finance', 'communications', 'research'],
            'direction': ['rally', 'decline', 'surge', 'fluctuate'],
            'news': ['economic data', 'policy changes', 'earnings reports', 'global events'],
            'action': ['surges', 'drops', 'stabilizes', 'fluctuates'],
            'event': ['regulatory news', 'market updates', 'economic reports'],
            'outcome': ['strong profits', 'steady growth', 'mixed results'],
            'change': ['increase', 'decrease', 'remain stable'],
            'trend': ['positive', 'concerning', 'mixed', 'encouraging'],
            'product': ['fund', 'service', 'platform', 'solution'],
            'market': ['stock prices', 'currency values', 'commodity prices'],
            'factor': ['geopolitical tensions', 'economic uncertainty', 'policy changes'],
            'discovery': ['species', 'planet', 'phenomenon', 'element'],
            'field': ['astronomy', 'biology', 'physics', 'chemistry'],
            'area': ['cancer research', 'climate science', 'space exploration'],
            'finding': ['concerning trends', 'positive developments', 'new insights'],
            'destination': ['Mars', 'deep space', 'the moon', 'asteroids'],
            'treatment': ['cancer therapy', 'genetic disorders', 'chronic diseases'],
            'result': ['alarming changes', 'positive trends', 'significant impacts'],
            'artifact': ['ancient ruins', 'historical artifacts', 'fossil remains'],
            'outcome': ['promising results', 'significant findings', 'breakthrough discoveries'],
            'project': ['climate research', 'space exploration', 'medical studies'],
            'condition': ['diabetes', 'heart disease', 'cancer', 'arthritis'],
            'statistic': ['declining rates', 'improving trends', 'concerning numbers'],
            'disease': ['flu', 'malaria', 'COVID-19', 'tuberculosis'],
            'issue': ['obesity', 'mental health', 'substance abuse', 'smoking'],
            'demographic': ['teenagers', 'elderly', 'working adults', 'students'],
            'activity': ['yoga', 'running', 'cycling', 'swimming'],
            'title': ['Blockbuster Hit', 'Indie Darling', 'Action Thriller', 'Romantic Comedy'],
            'name': ['Hollywood Star', 'Pop Icon', 'Rising Actor', 'Music Legend'],
            'project': ['album', 'movie', 'tour', 'collaboration'],
            'genre': ['rock', 'pop', 'jazz', 'electronic'],
            'award': ['Emmy', 'Oscar', 'Golden Globe', 'Critics Choice'],
            'play': ['Shakespeare classic', 'modern drama', 'musical hit'],
            'content': ['original series', 'exclusive movies', 'documentary'],
            'style': ['contemporary art', 'classical works', 'modern sculptures']
        }
    
    def generate_headline(self, category):
        """Generate a single headline for the given category"""
        template = random.choice(self.templates[category])
        
        # Replace placeholders with random fillers
        for placeholder, options in self.fillers.items():
            if f'{{{placeholder}}}' in template:
                template = template.replace(f'{{{placeholder}}}', random.choice(options))
        
        return template
    
    def generate_dataset(self, num_samples=1000):
        """Generate a dataset with specified number of samples"""
        headlines = []
        categories = []
        
        # Ensure balanced distribution across categories
        samples_per_category = num_samples // len(self.categories)
        remaining_samples = num_samples % len(self.categories)
        
        for i, category in enumerate(self.categories):
            # Add extra sample to first few categories if there's remainder
            category_samples = samples_per_category + (1 if i < remaining_samples else 0)
            
            for _ in range(category_samples):
                headlines.append(self.generate_headline(category))
                categories.append(category)
        
        # Shuffle the data
        combined = list(zip(headlines, categories))
        random.shuffle(combined)
        headlines, categories = zip(*combined)
        
        return pd.DataFrame({
            'Headline': headlines,
            'Category': categories
        })
    
    def save_dataset(self, filename='news_dataset.csv', num_samples=1000):
        """Generate and save dataset to CSV file"""
        df = self.generate_dataset(num_samples)
        df.to_csv(filename, index=False)
        print(f"Dataset with {num_samples} samples saved to {filename}")
        return df

if __name__ == "__main__":
    generator = NewsDataGenerator()
    df = generator.save_dataset(num_samples=1000)
    print(f"Generated dataset shape: {df.shape}")
    print(f"Category distribution:\n{df['Category'].value_counts()}")
