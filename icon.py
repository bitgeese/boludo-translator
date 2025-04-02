#!/usr/bin/env python3
import sys
import io
import emoji
from PIL import Image, ImageDraw, ImageFont
import cairosvg
import tempfile
import os
import requests

def emoji_to_favicon(emoji_str):
    """
    Convert an emoji string to a favicon.ico file
    
    Args:
        emoji_str (str): String containing emoji
    
    Returns:
        bytes: The favicon.ico file content as bytes
    """
    # Check if input contains emoji
    if not emoji.emoji_count(emoji_str):
        print("Error: No emoji found in input string")
        return None
    
    # Extract the first emoji from the string
    emoji_char = None
    for char in emoji_str:
        if emoji.is_emoji(char):
            emoji_char = char
            break
    
    if not emoji_char:
        print("Error: No valid emoji found")
        return None
    
    try:
        # Method 1: Try to render emoji using Pillow
        # Create a blank image with transparent background
        img_size = 256
        img = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to find a font that supports emoji
        try:
            # Try system fonts that might support emoji
            font_paths = [
                '/System/Library/Fonts/Apple Color Emoji.ttc',  # macOS
                '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',  # Linux with Noto
                'C:\\Windows\\Fonts\\seguiemj.ttf'  # Windows
            ]
            
            font = None
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, img_size // 2)  # Reduced size to avoid "invalid pixel size"
                        break
                    except Exception:
                        continue
            
            if font:
                # Calculate position to center the emoji
                try:
                    bbox = draw.textbbox((0, 0), emoji_char, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    # Fallback for older Pillow versions
                    w, h = draw.textsize(emoji_char, font=font)
                
                position = ((img_size - w) // 2, (img_size - h) // 2)
                
                # Draw the emoji
                draw.text(position, emoji_char, font=font, fill=(0, 0, 0, 255))
                
                # Check if the image is not empty (contains actual emoji rendering)
                if img.getbbox():
                    # Resize to favicon dimensions
                    favicon = img.resize((32, 32), Image.LANCZOS)
                    
                    # Save as ICO
                    ico_bytes = io.BytesIO()
                    favicon.save(ico_bytes, format='ICO', sizes=[(16, 16), (32, 32), (48, 48)])
                    return ico_bytes.getvalue()
        
        except Exception as e:
            print(f"Pillow rendering failed: {e}")
            # Fall back to method 2
            pass
        
        # Method 2: Use emoji SVG from Twemoji and convert to PNG then ICO
        try:
            # Get the Unicode code point(s) in the format needed for Twemoji
            code_points = '-'.join([f"{ord(c):x}" for c in emoji_char])
            
            # Twemoji URL
            twemoji_url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/svg/{code_points}.svg"
            
            response = requests.get(twemoji_url)
            if response.status_code == 200:
                # Convert SVG to PNG
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
                    png_path = png_file.name
                
                svg_data = response.content
                cairosvg.svg2png(bytestring=svg_data, write_to=png_path, output_width=256, output_height=256)
                
                # Open the PNG and convert to ICO
                with Image.open(png_path) as img:
                    # Resize to favicon dimensions
                    favicon = img.resize((32, 32), Image.LANCZOS)
                    
                    # Save as ICO
                    ico_bytes = io.BytesIO()
                    favicon.save(ico_bytes, format='ICO', sizes=[(16, 16), (32, 32), (48, 48)])
                    
                    # Clean up temporary file
                    os.unlink(png_path)
                    
                    return ico_bytes.getvalue()
            else:
                print(f"Failed to download emoji SVG: HTTP {response.status_code}")
        except Exception as e:
            print(f"SVG conversion failed: {e}")
        
        # Method 3: Create a colored square with the first letter of the emoji as fallback
        try:
            # Create a colored background
            img = Image.new('RGBA', (256, 256), (255, 165, 0, 255))  # Orange background
            draw = ImageDraw.Draw(img)
            
            # Try to find a standard font
            try:
                # Try to use a standard font
                standard_font_paths = [
                    '/System/Library/Fonts/Helvetica.ttc',  # macOS
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
                    'C:\\Windows\\Fonts\\arial.ttf'  # Windows
                ]
                
                std_font = None
                for path in standard_font_paths:
                    if os.path.exists(path):
                        std_font = ImageFont.truetype(path, 128)
                        break
                
                if std_font:
                    # Get the first character of the emoji name or the Unicode code point
                    emoji_name = emoji.demojize(emoji_char).replace(':', '').replace('_', ' ')
                    text = emoji_name[0].upper() if emoji_name else hex(ord(emoji_char))[2:4].upper()
                    
                    # Calculate position to center the text
                    try:
                        bbox = draw.textbbox((0, 0), text, font=std_font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except AttributeError:
                        w, h = draw.textsize(text, font=std_font)
                    
                    position = ((256 - w) // 2, (256 - h) // 2)
                    
                    # Draw the text
                    draw.text(position, text, font=std_font, fill=(255, 255, 255, 255))
            except Exception as e:
                print(f"Text rendering failed: {e}")
            
            # Resize to favicon dimensions
            favicon = img.resize((32, 32), Image.LANCZOS)
            
            # Save as ICO
            ico_bytes = io.BytesIO()
            favicon.save(ico_bytes, format='ICO', sizes=[(16, 16), (32, 32), (48, 48)])
            return ico_bytes.getvalue()
            
        except Exception as e:
            print(f"Fallback rendering failed: {e}")
            
            # Ultimate fallback: just a colored square
            img = Image.new('RGBA', (32, 32), (255, 0, 0, 255))
            ico_bytes = io.BytesIO()
            img.save(ico_bytes, format='ICO')
            return ico_bytes.getvalue()
    
    except Exception as e:
        print(f"Error creating favicon: {e}")
        return None

def main():
    if len(sys.argv) > 1:
        emoji_str = sys.argv[1]
    else:
        emoji_str = input("Enter an emoji: ")
    
    favicon_bytes = emoji_to_favicon(emoji_str)
    
    if favicon_bytes:
        # Write to file
        with open('favicon.ico', 'wb') as f:
            f.write(favicon_bytes)
        print(f"Favicon created successfully: favicon.ico")
    else:
        print("Failed to create favicon")

if __name__ == "__main__":
    main()
