#!/usr/bin/env python3
"""
LinkedIn Profile Roster Agent
Takes LinkedIn profile screenshots and generates smart roster suggestions with annotated images.
Uses Google Gemini LLM for profile analysis.
"""

import os
import sys
import time
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import json

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not installed. Run: pip install google-generativeai")
    sys.exit(1)


class LinkedInRosterAgent:
    """AI agent that analyzes LinkedIn profiles and generates roster suggestions."""
    
    def __init__(self, api_key: str = None):
        """Initialize the agent with Google Gemini API key."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass as argument.")
        
        genai.configure(api_key=self.api_key)
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception:
            try:
                self.model = genai.GenerativeModel('gemini-2.5-pro')
            except Exception:
                try:
                    self.model = genai.GenerativeModel('gemini-pro')
                except Exception:
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.roster_suggestions = []
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_profile(self, image_path: str) -> Dict:
        """Analyze LinkedIn profile screenshot using Google Gemini Vision API."""
        print(f"Analyzing LinkedIn profile from: {image_path}")
        
        img = Image.open(image_path)
        
        prompt = """You are a witty, humorous LinkedIn profile roaster who makes fun of profiles while giving genuinely helpful improvement suggestions. 
        Analyze this LinkedIn profile screenshot and ROAST it with humor, but also provide real improvement suggestions.

        TASKS:
        1. Extract basic information: name, title, company, location, education, skills, experience level
        2. Identify specific sections/blocks in the profile (e.g., headline, summary/about, experience entries, skills section, education section)
        3. For EACH identified section, provide:
           - Section name/type (e.g., "Headline", "About Section", "Experience Entry 1", "Skills Section")
           - Approximate location description (e.g., "top center", "middle left", "below name")
           - Current content assessment (be funny/roasting but constructive)
           - Humorous, roasting-style improvement suggestions (make fun of it but give real advice)
        4. Generate smart roster suggestions based on the profile

        IMPORTANT: Format your response as a valid JSON object with these exact keys:
        {
            "name": "extracted name",
            "title": "current job title",
            "company": "company name",
            "location": "location",
            "education": "education details",
            "skills": ["skill1", "skill2", "skill3"],
            "experience_level": "Junior/Mid/Senior/Executive",
            "roster_suggestions": ["suggestion1", "suggestion2", "suggestion3", "suggestion4", "suggestion5"],
            "sections": [
                {
                    "section_name": "Headline",
                    "location_description": "top center, below profile picture",
                    "current_content": "brief description of current content",
                    "improvements": ["funny roasting suggestion 1", "funny roasting suggestion 2"],
                    "priority": "high/medium/low"
                },
                {
                    "section_name": "About Section",
                    "location_description": "middle left, below headline",
                    "current_content": "brief description",
                    "improvements": ["funny roasting suggestion 1"],
                    "priority": "high/medium/low"
                }
            ]
        }

        Make the roasts EXTREME, SAVAGE, HILARIOUS, and BRUTALLY HONEST. Be ruthless but constructive. Examples:
        - "Your headline is a keyword vomit that screams 'I Googled how to get hired' - pick a personality!"
        - "This About section is giving 'I copy-pasted from ChatGPT' vibes. Where's the human?"
        - "Your skills section is the LinkedIn equivalent of listing 'breathing' as an achievement!"
        - "Your experience entries are so generic, they could be anyone's profile. Show me YOU!"

        IMPORTANT: Make roasts EXTREME and FUNNY - like a savage friend roasting you. Be brutal but helpful. Each roast should be 20-30 words and make people laugh while learning.

        Identify at least 3-5 key sections that could be improved. Focus on:
        - Headline/Title
        - About/Summary section
        - Experience entries
        - Skills section
        - Education section
        - Any other visible sections

        Return ONLY the JSON object, no additional text or markdown formatting."""
        
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content([prompt, img])
                content = response.text.strip()
                
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        profile_data = json.loads(content[json_start:json_end])
                    else:
                        profile_data = self._parse_text_response(content)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON response. Error: {e}")
                    print(f"Response content: {content[:200]}...")
                    profile_data = self._parse_text_response(content)
                
                break
                
            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        if "retry in" in error_str.lower():
                            try:
                                import re
                                match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                                if match:
                                    retry_delay = float(match.group(1)) + 1
                            except:
                                retry_delay = retry_delay * 2
                        
                        print(f"⚠️  Rate limit/quota exceeded. Retrying in {retry_delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        print(f"❌ Error: Quota/Rate limit exceeded. Please wait or check your API quota.")
                        print(f"   Free tier limit: 20 requests/day per model")
                        print(f"   Check usage: https://ai.dev/usage?tab=rate-limit")
                        raise Exception("API quota exceeded. Please wait before trying again or upgrade your API plan.")
                else:
                    print(f"Error calling Gemini API: {e}")
                    raise
        
        return profile_data
    
    def _parse_text_response(self, text: str) -> Dict:
        """Fallback parser for non-JSON responses."""
        return {
            "name": "Extracted from profile",
            "title": "Professional",
            "company": "Company",
            "location": "Location",
            "education": "Education",
            "skills": ["Skill1", "Skill2"],
            "experience_level": "Mid",
            "roster_suggestions": ["Technical Specialist", "Team Lead", "Senior Contributor"],
            "sections": [
                {
                    "section_name": "Headline",
                    "location_description": "top center",
                    "current_content": "Current headline",
                    "improvements": ["Your headline reads like a robot wrote it!"],
                    "priority": "medium"
                }
            ]
        }
    
    def generate_roster_suggestions(self, profile_data: Dict) -> List[str]:
        """Generate smart roster suggestions based on profile analysis."""
        suggestions = profile_data.get("roster_suggestions", [])
        
        if not suggestions:
            experience = profile_data.get("experience_level", "Mid").lower()
            title = profile_data.get("title", "").lower()
            
            if "junior" in experience or "entry" in experience:
                suggestions = ["Junior Contributor", "Associate", "Entry Level Specialist"]
            elif "senior" in experience or "executive" in experience or "director" in title or "vp" in title:
                suggestions = ["Senior Leader", "Executive Contributor", "Strategic Advisor", "Tech Lead"]
            else:
                suggestions = ["Mid-Level Specialist", "Team Contributor", "Technical Expert"]
        
        self.roster_suggestions = suggestions[:5]  # Limit to 5 suggestions
        return self.roster_suggestions
    
    def _get_section_position(self, location_desc: str, width: int, height: int) -> Tuple[int, int, int, int]:
        """Estimate section position based on location description."""
        desc_lower = location_desc.lower()
        
        # Estimate Y position
        if "top" in desc_lower:
            y_start = int(height * 0.05)
            y_end = int(height * 0.25)
        elif "middle" in desc_lower or "center" in desc_lower:
            y_start = int(height * 0.3)
            y_end = int(height * 0.6)
        elif "bottom" in desc_lower:
            y_start = int(height * 0.65)
            y_end = int(height * 0.95)
        else:
            y_start = int(height * 0.2)
            y_end = int(height * 0.5)
        
        if "left" in desc_lower:
            x_start = int(width * 0.05)
            x_end = int(width * 0.45)
        elif "right" in desc_lower:
            x_start = int(width * 0.55)
            x_end = int(width * 0.95)
        elif "center" in desc_lower or "middle" in desc_lower:
            x_start = int(width * 0.25)
            x_end = int(width * 0.75)
        else:
            x_start = int(width * 0.1)
            x_end = int(width * 0.6)
        
        return (x_start, y_start, x_end, y_end)
    
    def _get_text_color(self, img: Image.Image, x: int, y: int, width: int, height: int) -> Tuple[int, int, int]:
        """Determine contrasting text color based on background."""
        sample_size = min(50, width // 4, height // 4)
        x_sample = max(0, min(x, img.width - sample_size))
        y_sample = max(0, min(y, img.height - sample_size))
        
        if img.mode != 'RGB':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img
        
        total_brightness = 0
        sample_count = 0
        for dx in range(0, sample_size, 5):
            for dy in range(0, sample_size, 5):
                px = min(x_sample + dx, img_rgb.width - 1)
                py = min(y_sample + dy, img_rgb.height - 1)
                pixel = img_rgb.getpixel((px, py))
                brightness = (pixel[0] + pixel[1] + pixel[2]) / 3
                total_brightness += brightness
                sample_count += 1
        
        avg_brightness = total_brightness / sample_count if sample_count > 0 else 128
        
        if avg_brightness < 128:
            return (255, 255, 255)  # White text
        else:
            return (0, 0, 0)  # Black text
    
    def _draw_text_with_outline(self, draw: ImageDraw.Draw, x: int, y: int, text: str, 
                                 font: ImageFont.FreeTypeFont, fill_color: Tuple[int, int, int],
                                 outline_color: Tuple[int, int, int] = (0, 0, 0), outline_width: int = 2):
        """Draw text with outline for better visibility."""
        for adj in range(-outline_width, outline_width + 1):
            for adj2 in range(-outline_width, outline_width + 1):
                if adj != 0 or adj2 != 0:
                    draw.text((x + adj, y + adj2), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=fill_color)
    
    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.Draw) -> List[str]:
        """Wrap text to fit within max_width, ensuring complete words."""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _draw_speech_bubble(self, draw: ImageDraw.Draw, x: int, y: int, width: int, height: int, 
                           fill_color: Tuple[int, int, int, int], outline_color: Tuple[int, int, int],
                           pointer_x: int, pointer_y: int, pointer_direction: str = "down"):
        """Draw a speech bubble with a pointer arrow."""
        draw.rectangle(
            [(x, y), (x + width, y + height)],
            fill=fill_color,
            outline=outline_color,
            width=4
        )
        
        pointer_size = 20
        if pointer_direction == "down":
            points = [
                (pointer_x, y + height),
                (pointer_x - pointer_size, y + height + pointer_size),
                (pointer_x + pointer_size, y + height + pointer_size)
            ]
        elif pointer_direction == "up":
            points = [
                (pointer_x, y),
                (pointer_x - pointer_size, y - pointer_size),
                (pointer_x + pointer_size, y - pointer_size)
            ]
        elif pointer_direction == "left":
            points = [
                (x, pointer_y),
                (x - pointer_size, pointer_y - pointer_size),
                (x - pointer_size, pointer_y + pointer_size)
            ]
        else:
            points = [
                (x + width, pointer_y),
                (x + width + pointer_size, pointer_y - pointer_size),
                (x + width + pointer_size, pointer_y + pointer_size)
            ]
        
        draw.polygon(points, fill=fill_color, outline=outline_color, width=4)
    
    def annotate_image(self, image_path: str, profile_data: Dict, output_path: str):
        """Annotate with COMPLETE, VISIBLE roasts + improvements + roster for each section."""
        print(f"Creating complete roasts with improvements and roster suggestions...")
        
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        try:
            font_paths = [
                "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
                "/System/Library/Fonts/Supplemental/Marker Felt.ttf",
                "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
            large_font = None
            medium_font = None
            small_font = None
            
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        large_font = ImageFont.truetype(path, 48)
                        medium_font = ImageFont.truetype(path, 40)
                        small_font = ImageFont.truetype(path, 36)
                        break
                    except:
                        continue
            
            if not large_font:
                large_font = ImageFont.load_default()
                medium_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        except:
            large_font = ImageFont.load_default()
            medium_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        width, height = img.size
        
        sections = profile_data.get("sections", [])
        roster_suggestions = self.roster_suggestions or profile_data.get("roster_suggestions", [])
        
        if not sections:
            sections = [
                {
                    "section_name": "Headline",
                    "location_description": "top center",
                    "improvements": ["Your headline reads like a robot wrote it! Add personality and be specific."],
                    "priority": "high"
                },
                {
                    "section_name": "About",
                    "location_description": "middle left",
                    "improvements": ["This About section is drier than the Sahara! Tell a story with real examples."],
                    "priority": "high"
                },
                {
                    "section_name": "Skills",
                    "location_description": "middle right",
                    "improvements": ["Skills? More like 'I Googled these keywords'! Be specific and show expertise."],
                    "priority": "medium"
                }
            ]
        
        print(f"Roasting {len(sections)} sections with COMPLETE text...")
        
        color_schemes = {
            "high": {
                "roast_color": (255, 0, 0),  # Bright Red for roast
                "improve_color": (0, 150, 0),  # Green for improvement
                "roster_color": (255, 140, 0),  # Orange for roster
                "outline": (0, 0, 0),  # Black outline for visibility
            },
            "medium": {
                "roast_color": (255, 69, 0),  # Red Orange for roast
                "improve_color": (0, 128, 128),  # Teal for improvement
                "roster_color": (138, 43, 226),  # Blue Violet for roster
                "outline": (0, 0, 0),  # Black outline
            },
            "low": {
                "roast_color": (255, 165, 0),  # Orange for roast
                "improve_color": (46, 139, 87),  # Sea Green for improvement
                "roster_color": (30, 144, 255),  # Dodger Blue for roster
                "outline": (0, 0, 0),  # Black outline
            }
        }
        
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        line_height = 50
        current_y_positions = {}
        
        for idx, section in enumerate(sections):
            section_name = section.get("section_name", f"Section {idx+1}")
            location_desc = section.get("location_description", "middle")
            improvements = section.get("improvements", [])
            priority = section.get("priority", "medium").lower()
            
            if not improvements:
                continue
            
            x_start, y_start, x_end, y_end = self._get_section_position(location_desc, width, height)
            section_center_x = (x_start + x_end) // 2
            section_center_y = (y_start + y_end) // 2
            
            scheme = color_schemes.get(priority, color_schemes["medium"])
            
            roast_text = improvements[0] if improvements else "Needs improvement!"
            improve_text = improvements[1] if len(improvements) > 1 else "Focus on specific achievements and metrics."
            
            roster_idx = idx % len(roster_suggestions) if roster_suggestions else 0
            roster_text = roster_suggestions[roster_idx] if roster_suggestions else f"{section_name} Roster"
            
            if idx % 4 == 0:
                text_x = max(30, x_start - 50)
                text_y = max(40, y_start - 180)
                arrow_start_x = text_x + 200
                arrow_start_y = text_y + 60
                arrow_end_x = x_start
                arrow_end_y = y_start
            elif idx % 4 == 1:
                text_x = min(width - 400, x_end + 30)
                text_y = max(40, y_start - 180)
                arrow_start_x = text_x
                arrow_start_y = text_y + 60
                arrow_end_x = x_end
                arrow_end_y = y_start
            elif idx % 4 == 2:
                text_x = max(30, x_start - 50)
                text_y = min(height - 200, y_end + 40)
                arrow_start_x = text_x + 200
                arrow_start_y = text_y
                arrow_end_x = x_start
                arrow_end_y = y_end
            else:
                text_x = min(width - 400, x_end + 30)
                text_y = min(height - 200, y_end + 40)
                arrow_start_x = text_x
                arrow_start_y = text_y
                arrow_end_x = x_end
                arrow_end_y = y_end
            
            text_x = max(20, min(text_x, width - 380))
            text_y = max(30, min(text_y, height - 180))
            
            overlay_draw.line(
                [(arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y)],
                fill=(0, 0, 0),
                width=3
            )
            
            # Draw arrowhead
            arrow_size = 10
            dx = arrow_end_x - arrow_start_x
            dy = arrow_end_y - arrow_start_y
            if abs(dx) > abs(dy):
                if dx > 0:
                    arrow_points = [(arrow_end_x, arrow_end_y), 
                                  (arrow_end_x - arrow_size, arrow_end_y - arrow_size),
                                  (arrow_end_x - arrow_size, arrow_end_y + arrow_size)]
                else:
                    arrow_points = [(arrow_end_x, arrow_end_y),
                                  (arrow_end_x + arrow_size, arrow_end_y - arrow_size),
                                  (arrow_end_x + arrow_size, arrow_end_y + arrow_size)]
            else:
                if dy > 0:
                    arrow_points = [(arrow_end_x, arrow_end_y),
                                  (arrow_end_x - arrow_size, arrow_end_y - arrow_size),
                                  (arrow_end_x + arrow_size, arrow_end_y - arrow_size)]
                else:
                    arrow_points = [(arrow_end_x, arrow_end_y),
                                  (arrow_end_x - arrow_size, arrow_end_y + arrow_size),
                                  (arrow_end_x + arrow_size, arrow_end_y + arrow_size)]
            
            overlay_draw.polygon(arrow_points, fill=(0, 0, 0))
            
            # Draw section name
            section_label = f"📍 {section_name}:"
            self._draw_text_with_outline(
                overlay_draw, text_x, text_y, section_label,
                font=medium_font, fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=3
            )
            text_y += line_height
            
            # Draw ROAST in red/orange color
            roast_label = f"💀 ROAST:"
            self._draw_text_with_outline(
                overlay_draw, text_x, text_y, roast_label,
                font=medium_font, fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=2
            )
            text_y += line_height
            
            roast_lines = self._wrap_text(roast_text, small_font, 350, overlay_draw)
            for line in roast_lines:
                self._draw_text_with_outline(
                    overlay_draw, text_x + 20, text_y, line,
                    font=small_font, fill_color=scheme['roast_color'], outline_color=(255, 255, 255), outline_width=2
                )
                text_y += line_height - 5
            
            text_y += 10
            
            # Draw IMPROVE in green color
            improve_label = f"💡 IMPROVE:"
            self._draw_text_with_outline(
                overlay_draw, text_x, text_y, improve_label,
                font=medium_font, fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=2
            )
            text_y += line_height
            
            improve_lines = self._wrap_text(improve_text, small_font, 350, overlay_draw)
            for line in improve_lines:
                self._draw_text_with_outline(
                    overlay_draw, text_x + 20, text_y, line,
                    font=small_font, fill_color=scheme['improve_color'], outline_color=(255, 255, 255), outline_width=2
                )
                text_y += line_height - 5
            
            text_y += 10
            
            # Draw ROSTER in blue/orange color
            roster_label = f"🎯 ROSTER:"
            self._draw_text_with_outline(
                overlay_draw, text_x, text_y, roster_label,
                font=medium_font, fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=2
            )
            text_y += line_height
            
            roster_line = roster_text
            self._draw_text_with_outline(
                overlay_draw, text_x + 20, text_y, roster_line,
                font=small_font, fill_color=scheme['roster_color'], outline_color=(255, 255, 255), outline_width=2
            )
        
        # Merge overlay
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        
        # Save
        img.save(output_path, quality=95)
        print(f"Complete roasted image saved!")
        print(f"  - {len(sections)} sections with roasts + improvements + roster")
        
        return img
    
    def process_profile(self, input_image_path: str, output_image_path: str = None) -> Dict:
        """Main method to process a LinkedIn profile screenshot."""
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Image not found: {input_image_path}")
        
        # Analyze profile
        profile_data = self.analyze_profile(input_image_path)
        print(f"\nProfile Analysis Complete:")
        print(f"Name: {profile_data.get('name', 'N/A')}")
        print(f"Title: {profile_data.get('title', 'N/A')}")
        print(f"Company: {profile_data.get('company', 'N/A')}")
        
        # Display sections identified
        sections = profile_data.get('sections', [])
        if sections:
            print(f"\n📋 Sections Identified ({len(sections)}):")
            for idx, section in enumerate(sections, 1):
                print(f"  {idx}. {section.get('section_name', 'Unknown')} ({section.get('priority', 'medium')} priority)")
                improvements = section.get('improvements', [])
                if improvements:
                    print(f"     Improvements: {', '.join(improvements[:2])}")
        
        # Generate roster suggestions
        suggestions = self.generate_roster_suggestions(profile_data)
        print(f"\n🎯 Roster Suggestions:")
        for idx, suggestion in enumerate(suggestions, 1):
            print(f"  {idx}. {suggestion}")
        
        # Generate output path if not provided
        if not output_image_path:
            base_name = os.path.splitext(os.path.basename(input_image_path))[0]
            output_dir = os.path.dirname(input_image_path) or "."
            output_image_path = os.path.join(output_dir, f"{base_name}_rostered.jpg")
        
        # Annotate image
        self.annotate_image(input_image_path, profile_data, output_image_path)
        
        return {
            "profile_data": profile_data,
            "roster_suggestions": suggestions,
            "output_image": output_image_path
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LinkedIn Profile Roster Agent - Analyze profiles and generate roster suggestions"
    )
    parser.add_argument(
        "input_image",
        help="Path to LinkedIn profile screenshot"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for annotated image (default: input_image_rostered.jpg)",
        default=None
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (or set GEMINI_API_KEY env variable)",
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        # Use provided API key or default from environment
        api_key = args.api_key or os.getenv("GEMINI_API_KEY") or "AIzaSyAHUpx_fWs5cqlHs8DHbQYMtColfoAVRoY"
        agent = LinkedInRosterAgent(api_key=api_key)
        result = agent.process_profile(args.input_image, args.output)
        
        print(f"\n✅ Success! Annotated image saved to: {result['output_image']}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()




