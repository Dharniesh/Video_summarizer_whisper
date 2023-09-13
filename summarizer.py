import streamlit as st
import openai
import whisper
import subprocess
import tiktoken
import os
import tempfile

openai.api_key=st.secrets['openai_api']

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

def chunker(text, max_tokens=14000):
    token_count = len(encoding.encode(text))
    return (token_count + max_tokens - 1) // max_tokens

def split_string(string, n):
    part_length = len(string) // n
    return [string[i:i + part_length] for i in range(0, len(string), part_length)]

def summarize_aud(chun_txt):
    prompt1 = """
    You are a helpful assistant that summarizes videos.
    You are provided chunks of raw audio that were transcribed from the video's audio.
    Summarize the current chunk to succinct and clear bullet points of its contents.
    """

    prompt = prompt1 + chun_txt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=700,
        temperature=0.7,
    )
    return response.choices[0].message.content

def mp42wav(input_file, output_file="audio_file.mp3"):
    command = [
        "ffmpeg", "-i", input_file, "-y", "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", output_file
    ]
    subprocess.call(command)
    return output_file

def wav2txt(output_file):
    model = whisper.load_model("base")
    result = model.transcribe(output_file, language='en')
    with open("transcription.txt", "w", encoding="utf-8") as txt:
        txt.write(result["text"])
    with open('transcription.txt', 'r') as file:
        trans = file.read()
    return trans

def main():
    st.title("Video Summarizer")
    st.write("Upload a video file to generate a summary.")

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.write("Video uploaded successfully!")

        # Create a temporary directory to save the uploaded video
        temp_dir = tempfile.mkdtemp()
        file_name = os.path.join(temp_dir, "uploaded_video.mp4")

        # Save the uploaded video to the temporary directory
        with open(file_name, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        output_file = mp42wav(file_name)
        trans = wav2txt(output_file)
        n = chunker(trans)
        result = split_string(trans, n)

        summarized_txt = ''

        for chun_txt in result:
            summarized_txt += summarize_aud(chun_txt)

        with open('summary.txt', 'w') as file:
            file.write(summarized_txt)

        st.write("Summary Generated:")
        st.write(summarized_txt)
        st.write(f"Number of chunks is {n}")

if __name__ == "__main__":
    main()
