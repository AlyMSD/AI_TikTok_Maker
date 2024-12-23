import csv
import os
import tempfile
import numpy as np
from moviepy.editor import *
from pyt2s.services import stream_elements
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionXLPipeline
import time


class YouTubeShortsGenerator:
    def __init__(self, csv_file, output_dir, background_image_path='bg.png'):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.temp_dir = tempfile.mkdtemp()
        self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        self.vote_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 80)  # Larger font for vote count
        self.image_size_ratio = 0.5  # 50% of original size
        self.background_image_path = background_image_path

        # Load Stable Diffusion XL model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "etri-vilab/koala-700m-llava-cap", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def parse_csv(self):
        questions = []

        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                questions.append({
                    'question_a': row[0],
                    'votes_a': int(row[1]),
                    'image_search_keywords_a': row[2],
                    'question_b': row[3],
                    'votes_b': int(row[4]),
                    'image_search_keywords_b': row[5]
                })
        return questions

    def generate_image(self, prompt):
        """Generate an image using Stable Diffusion XL based on the provided prompt."""
        negative_prompt = "distorted, abstract, ugly, worst quality, low quality, illustration, low resolution"
        image = self.pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
        new_size = (900, 900)  # Resize generated image
        resized_img = image.resize(new_size)
        return resized_img

    def create_tts_audio(self, text, filename):
        data = stream_elements.requestTTS(text, stream_elements.Voice.Matthew.value)
        with open('output.mp3', '+wb') as file:
            file.write(data)
        return AudioFileClip('output.mp3')

    def draw_text_wrapped(self, draw, text, position, max_width, fill, font=None):
        if font is None:
            font = self.font
        lines = []
        words = text.split()
        line = ''
        for word in words:
            test_line = f'{line} {word}'.strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        lines.append(line)

        y_offset = position[1] - (len(lines) * font.size) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            draw.text((position[0] - width // 2, y_offset), line, font=font, fill=fill)
            y_offset += font.size

    def create_video_segment(self, question_data):
        print(f"Creating video segment for: {question_data['question_a']} vs {question_data['question_b']}")

        # Generate images using Stable Diffusion XL
        img_a = self.generate_image(question_data['image_search_keywords_a'])
        img_b = self.generate_image(question_data['image_search_keywords_b'])

        # Create the initial question image
        img = self.create_question_image(
            question_data['question_a'],
            img_a,
            question_data['question_b'],
            img_b
        )

        # Pre-compute the vote image
        img_with_votes = self.create_question_image("", img_a, "", img_b)
        draw = ImageDraw.Draw(img_with_votes)
        votes = question_data['votes_a'] + question_data['votes_b']

        percent_a = round((question_data['votes_a'] / votes) * 100)
        percent_b = round((question_data['votes_b'] / votes) * 100)

        vote_data = [percent_a, percent_b]
        position_data = [(540, int(360 * 0.7)), (540, int(1320 * 0.9))]
        self.replace_question_with_vote(draw, vote_data, position=position_data)  # Centered

        # Convert PIL Images to numpy arrays
        img_array = np.array(img)
        img_with_votes_array = np.array(img_with_votes)

        # Get duration of audio clips
        audio_a = self.create_tts_audio(question_data['question_a'] + " Or", os.path.join(self.temp_dir, 'audio_a.mp3'))
        audio_b = self.create_tts_audio(question_data['question_b'], os.path.join(self.temp_dir, 'audio_b.mp3'))
        ding_sound = AudioFileClip('ding.mp3')  # Load the ding sound
        audio_duration = audio_a.duration + audio_b.duration

        def add_votes(t):
            if t < audio_duration:  # Show the question images while TTS is playing
                return img_array
            elif t >= audio_duration and t < audio_duration + 3:  # Show the vote image after TTS is done
                return img_with_votes_array
            else:  # After votes are shown, return a blank image (optional)
                return img_with_votes_array  # Or any other image you want to show

        video = VideoClip(add_votes, duration=audio_duration + 2)  # Extend duration for ding sound

        audio = concatenate_audioclips(
            [audio_a, audio_b.set_start(audio_a.duration), ding_sound.set_start(audio_duration + 3)]
        )
        video = video.set_audio(audio)

        return video

    def create_question_image(self, question_a, img_a, question_b, img_b):
        # Open the background image
        background_img = Image.open(self.background_image_path).resize((1080, 1920))

        # Resize images to 50% of original size
        img_a = img_a.resize((int(img_a.width * self.image_size_ratio), int(img_a.height * self.image_size_ratio)),
                             Image.LANCZOS)
        img_b = img_b.resize((int(img_b.width * self.image_size_ratio), int(img_b.height * self.image_size_ratio)),
                             Image.LANCZOS)

        # Create a blank canvas for the combined image
        img_combined = Image.new('RGB', (1080, 1920))
        img_combined.paste(background_img, (0, 0))  # Use background

        # Center images in their respective halves
        question_a_y = int(360 * 0.7) + 40  # Position for img_a below question_a with a 40px buffer
        question_b_y = int(1320 * 0.9) + 40  # Position for img_b below question_b with a 40px buffer

        img_combined.paste(img_a, (540 - img_a.width // 2, question_a_y))  # Center img_a in the top half
        img_combined.paste(img_b, (540 - img_b.width // 2, question_b_y))  # Center img_b in the bottom half

        draw = ImageDraw.Draw(img_combined)

        if question_a:
            self.draw_text_wrapped(draw, question_a, (540, int(360 * 0.7)), 1000, 'white')
        if question_b:
            self.draw_text_wrapped(draw, question_b, (540, int(1320 * 0.9)), 1000, 'white')

        return img_combined

    def replace_question_with_vote(self, draw, votes, position):
        if votes[0] > votes[1]:
            vote_large = f"{votes[0]}%"
            self.draw_text_wrapped(draw, vote_large, position[0], max_width=1000, fill='#3efa34', font=self.vote_font)
            vote_small = f"{votes[1]}%"
            self.draw_text_wrapped(draw, vote_small, position[1], max_width=1000, fill='white', font=self.vote_font)
        else:
            vote_large = f"{votes[1]}%"
            self.draw_text_wrapped(draw, vote_large, position[1], max_width=1000, fill='#3efa34', font=self.vote_font)
            vote_small = f"{votes[0]}%"
            self.draw_text_wrapped(draw, vote_small, position[0], max_width=1000, fill='white', font=self.vote_font)


    def get_expected_duration(self, question_data):
        # Estimate duration based on TTS audio duration
        audio_a = self.create_tts_audio(question_data['question_a'] + " Or", os.path.join(self.temp_dir, 'audio_a.mp3'))
        audio_b = self.create_tts_audio(question_data['question_b'], os.path.join(self.temp_dir, 'audio_b.mp3'))

        # The total expected duration is the duration of both audio clips plus any additional time (e.g., for the vote display)
        total_duration = audio_a.duration + audio_b.duration + 2  # Add extra time for vote display
        return total_duration


    def create_full_video(self, questions):
        segments = []
        total_duration = 0

        for q in questions[:12]:
            # Calculate the expected duration based on the text-to-speech duration
            expected_duration = self.get_expected_duration(q)

            # Check if adding this segment would exceed the 60-second limit
            if total_duration + expected_duration <= 58:
                segment = self.create_video_segment(q)
                segments.append(segment)
                total_duration += expected_duration
            else:
                print(f"Skipping question: '{q['question_a']}' vs '{q['question_b']}' to stay within 60 seconds.")
                break  # Stop processing more questions if the time limit is exceeded

        if len(segments) == 0:
            raise ValueError("No segments available under 58 seconds.")

        final_video = concatenate_videoclips(segments, method="compose")
        return final_video

    def generate_video(self):
        questions = self.parse_csv()
        video = self.create_full_video(questions)
        output_path = os.path.join(self.output_dir, 'static/videos/youtube_shorts.mp4')
        video.write_videofile(output_path, fps=30)
        video = VideoFileClip(output_path)

        # Write the video to a MOV file
        video.write_videofile(output_path[:-4] + 'MOV.mov', codec="libx264")
        print(f"Video saved to {output_path}")
        return output_path
        # return False


