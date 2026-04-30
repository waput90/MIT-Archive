#!/usr/bin/env python3
"""
Audio Data Collection and Transcript Creation Tools (Cebuano Version)
Helps you record audio samples and create transcript files for voice cloning training.
Uses Cebuano language sample texts.
"""

import os
import sys
import time
import json
import wave
import pyaudio
import threading
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from datetime import datetime

REQUIRED_CONTRIBUTOR_ACK = """
CONTRIBUTOR POLICY ACKNOWLEDGEMENT

By contributing audio/transcripts to this repository, you confirm:
1) You have read the repository policy documents.
2) Your contributed audio/transcripts may be released for free non-commercial research and educational use.
3) Commercial use of contributed dataset material is not allowed in this repository.
4) This main repository is scoped to Davao Cebuano + Whisper workflows.
5) Work for other Philippine languages or non-Whisper models must be done in a fork.
6) You agree to comply with Philippine privacy law, including Republic Act No. 10173 (Data Privacy Act of 2012).
7) Public voice cloning, speaker impersonation, or synthetic voice generation of contributed speakers is prohibited.

Type I AGREE to continue.
"""


def require_contributor_acknowledgement(assume_yes: bool = False) -> bool:
    """Require explicit acknowledgement before collecting or adding data."""
    print("\n" + "=" * 72)
    print(REQUIRED_CONTRIBUTOR_ACK.strip())
    print("=" * 72)

    if assume_yes:
        print("Policy acknowledgement accepted via --agree-policy flag.")
        return True

    response = input("\nType I AGREE to continue: ").strip().upper()
    if response != "I AGREE":
        print("Policy acknowledgement not provided. Exiting without recording.")
        return False

    return True


# Try to import required libraries
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install sounddevice soundfile pyaudio")
    sys.exit(1)

class AudioRecorder:
    """
    Class for recording high-quality audio samples
    """
    
    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_data = []
        
    def list_audio_devices(self):
        """List available audio input devices"""
        print("Available audio devices:")
        print("-" * 40)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']} (inputs: {device['max_input_channels']})")
        print("-" * 40)
    
    def record_audio(self, duration: float = None, device_id: int = None) -> np.ndarray:
        """
        Record audio for specified duration or until stopped
        """
        print(f"Recording audio... Sample rate: {self.sample_rate}Hz")
        if duration:
            print(f"Duration: {duration} seconds")
        else:
            print("Press Ctrl+C to stop recording")
        
        try:
            if duration:
                # Record for specific duration
                audio_data = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    device=device_id,
                    dtype=np.float32
                )
                sd.wait()  # Wait until recording is finished
            else:
                # Record until interrupted
                audio_data = []
                self.is_recording = True
                
                def callback(indata, frames, time, status):
                    if self.is_recording:
                        audio_data.extend(indata.copy())
                
                with sd.InputStream(
                    callback=callback,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    device=device_id,
                    dtype=np.float32
                ):
                    while self.is_recording:
                        time.sleep(0.1)
                
                audio_data = np.array(audio_data).reshape(-1, self.channels)
        
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
            self.is_recording = False
            if 'audio_data' in locals() and isinstance(audio_data, list):
                audio_data = np.array(audio_data).reshape(-1, self.channels)
        
        print("Recording complete!")
        return audio_data
    
    def save_audio(self, audio_data: np.ndarray, filename: str):
        """Save audio data to file"""
        sf.write(filename, audio_data, self.sample_rate)
        print(f"Audio saved to: {filename}")

class DatasetCreator:
    """
    Create structured dataset for voice cloning training
    """
    
    def __init__(self, dataset_dir: str = "audio_data"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        self.metadata_file = self.dataset_dir / "metadata.json"
        self.transcript_file = self.dataset_dir / "transcripts.txt"
        
        # Load existing metadata if available
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> Dict:
        """Load existing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"recordings": [], "total_duration": 0, "total_files": 0}
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def add_recording(self, audio_data: np.ndarray, text: str, speaker_name: str = "default"):
        """Add a new recording to the dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{speaker_name}_{timestamp}.wav"
        filepath = self.dataset_dir / filename
        
        # Save audio file
        sf.write(filepath, audio_data, 22050)
        
        # Calculate duration
        duration = len(audio_data) / 22050
        
        # Add to metadata
        recording_info = {
            "filename": filename,
            "text": text,
            "speaker": speaker_name,
            "duration": duration,
            "timestamp": timestamp,
            "sample_rate": 22050,
            "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
        }
        
        self.metadata["recordings"].append(recording_info)
        self.metadata["total_duration"] += duration
        self.metadata["total_files"] += 1
        
        # Save metadata
        self.save_metadata()
        
        # Update transcript file
        self.update_transcript_file()
        
        print(f"Added recording: {filename} ({duration:.2f}s)")
        return filename
    
    def update_transcript_file(self):
        """Update the transcript file in the format expected by the voice cloning script"""
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            for recording in self.metadata["recordings"]:
                # Format: filename_without_extension|text
                filename_base = Path(recording["filename"]).stem
                f.write(f"{filename_base}|{recording['text']}\n")
    
    def get_dataset_stats(self):
        """Get statistics about the current dataset"""
        stats = {
            "total_files": self.metadata["total_files"],
            "total_duration": self.metadata["total_duration"],
            "average_duration": self.metadata["total_duration"] / max(1, self.metadata["total_files"]),
            "speakers": list(set(r["speaker"] for r in self.metadata["recordings"]))
        }
        return stats

class InteractiveRecorder:
    """
    Interactive recording session for creating training data
    """
    
    def __init__(self, dataset_dir: str = "audio_data"):
        self.recorder = AudioRecorder()
        self.dataset = DatasetCreator(dataset_dir)
        self.sample_texts = self.load_sample_texts()
    
    def load_sample_texts(self) -> List[str]:
        """Load sample texts for recording in Cebuano"""
        # Cebuano phrases covering various phonemes and common expressions
        texts = [
            "Ang paspas nga kayumanggi nga fox kay mi-ambak sa tamad na iro.",
            "Maayong buntag, kamusta imong adlaw?",
            "Ang teknolohiya nagabag-o sa atong kalibutan kada adlaw.",
            "Ganahan ko mag basa ug mga libro ug mag tu-on ug bag-ong kaalam.",
            "Ang musika adunay gahum nga makapalipay sa atong kasing-kasing ug kalag.",
            "Salamat sa imong tabang ug sa imong malasakit sa ako.",
            "Ang panahon karon kay nindut ug naay kahayag.",
            "Ang pamilya ug mga higala importante sa kinabuhi.",
            "Ga suporta kami sa usa ka kanunayng adhikain.",
            "Ang edukasyon susi sa pag-asenso.",
            "Mi-adto kami sa merkado para mamalit ug mga pagkaon.",
            "Ang kaligayahan at pagmamahal ay nagpapayaman sa ating buhay.",
            "Magtrabaho tayo nang may sipag at integridad.",
            "Ang kalikasan ay nagbibigay sa atin ng kagandahan at inspirasyon.",
            "Magkaroon kami ng malaking pangarap at layunin sa hinaharap.",
            "Ang kapakanan ng bawat tao ay mahalaga sa lipunan.",
            "Magsama tayo at magdiwang ng ating tagumpay.",
            "Ang kabaitan at malasakit ay pundasyon ng magandang mundo.",
            "Makinig nang mabuti sa mga payo ng ating mga nakatatanda.",
            "Ang buhay ay puno ng pagkakataon at hamon na dapat nating harapin."
        ]

        additional_texts = [
            "Karong buntaga, nagluto si mama ng adobo while I prepared the coffee.",
            "Please ayaw kalimti ang keys kay late na tayo for class.",
            "Niadto mi sa market kanina, and we bought fresh fish and gulay.",
            "Kung human na ka, paki-send na lang the final report tonight.",
            "Mag practice ta ug pronunciation para mas clear ang recording quality.",
            "Sa office kahapon, everyone was busy pero chill pa rin ang vibe.",
            "Dali ra ni nga task, just follow the steps and click submit.",
            "Nindot ang weather today, kaya tara mag walking after lunch.",
            "Nag review ko sa notes kagabi, then I slept early para fresh.",
            "Palihug check the microphone settings before ta mag start recording.",
            "Kung traffic sa highway, mag alternate route na lang tayo via downtown.",
            "Naay meeting ugma, so prepare your slides and speaking points.",
            "Ganina sa school, nag demo ang teacher about data preprocessing.",
            "Ayaw kabalaka, we can fix the bug after a quick break.",
            "Nag order mi og lunch online, then the rider arrived on time.",
            "Please i-save ang file sa correct folder para dili mawala.",
            "Mas maayo kung mag warm-up ka before speaking para clear ang voice.",
            "Kanang sentence pwede nimo usbon, but keep the same meaning please.",
            "Naka human ko sa assignment, kaya makatulog na ako nang maaga.",
            "Pag abot sa station, kindly wait near the blue sign.",
            "Ibutang ang date sa taas, then write your name below.",
            "Sa training session, nag share siya ng tips for better diction.",
            "Dili kinahanglan perfect, just be consistent sa imong pacing.",
            "Mamaya after dinner, mag record ta ug five more samples.",
            "Nag enjoy mi sa workshop kay practical ang examples and activities.",
            "Kung low ang battery, i-charge dayon before the next take.",
            "Paki-repeat ang phrase slowly para accurate ang transcription.",
            "Naa koy small update, we moved the deadline to Friday.",
            "Karong hapon, mag clean ta sa dataset and remove noisy clips.",
            "Ayaw saba sa background, because the model needs clean audio.",
            "Nipalit ko ng notebook and pens para sa field notes tomorrow.",
            "Pag human sa call, i-upload dayon ang outputs sa drive.",
            "Mao ni ang final version, please review before we publish.",
            "Nag ask si sir kung ready na ba ang experiment setup.",
            "Kung okay na ang levels, start na ta sa first sentence.",
            "I-confirm lang sa group chat if naka receive ka sa file.",
            "Sige lang, we can iterate until maayo na gyud ang result.",
            "Sa hallway earlier, we discussed the annotation guidelines.",
            "Magdala ko ug extra water kay dugay ang recording session.",
            "Please keep your phone silent para walay interruption.",
            "Nindot paminawon ang sample, pero adjust gamay ang gain.",
            "Mamili ka ng topic, then explain it in one minute.",
            "Nag suggest siya nga mag benchmark ta against the baseline model.",
            "Ayaw ka pressure, just speak naturally and smile while reading.",
            "Pag ready na ka, press record and count to three.",
            "Sa cafeteria kanina, we planned the next sprint tasks.",
            "Mura og lisod sa una, but it gets easier with practice.",
            "I-note ang pronunciation variants para useful sa evaluation.",
            "Karong weekend, mag collect pa ta ug additional voice data.",
            "Please label each file correctly before uploading to the repo.",
            "Naka set na ang timer, so start speaking when you hear the beep.",
            "Kung naay error message, screenshot it and send sa team.",
            "Nag ulan ganiha, but we still made it to the lab.",
            "Ayaw pagdali, enunciate each word for better recognition.",
            "Paki-close ang door para less ang ambient noise.",
            "Sa module nato today, we cover cleaning, splitting, and validation.",
            "Naa sa checklist ang consent form, please sign before recording.",
            "I-update ang metadata kung kinsa ang speaker ug unsay prompt.",
            "Kanus-a ang demo day, and who will present the findings.",
            "Mag meeting ta later to finalize the publication draft.",
            "Naglibog ko gamay sa format, pero clear na after your explanation.",
            "Dili tanan clips usable, so filter out those with distortions.",
            "Please speak one sentence at a time, then pause for two seconds.",
            "Sa jeep pauwi, I reviewed the transcript on my phone.",
            "I-verify nato ang sample rate para consistent ang training input.",
            "Kung kulang pa ang data, add ten more recordings tonight.",
            "Naa sa README ang instructions, just follow them step by step.",
            "Paki-check ang spelling kay importante sa language modeling.",
            "Pag abot sa office, nag set up dayon mi sa equipment.",
            "Mas paspas ang workflow kung organized ang folders and filenames.",
            "Karong adlawa, mag focus ta sa Cebuano-Tagalog mixed prompts.",
            "Please avoid whispering kay kinahanglan steady and audible ang voice.",
            "Nindot unta kung naay quiet room for the next batch.",
            "Nag test mi sa new mic, and the clarity improved a lot.",
            "Ibutang sa notes kung unsang words ang lisod i-pronounce.",
            "Kung human na ang run, export the logs and metrics.",
            "Sa presentation bukas, include examples of code-switched utterances.",
            "Ayaw kalimot mag backup sa files before any major edit.",
            "Nag ask si maam for a short summary and action items.",
            "Puwede ta mag pair work para mas dali ang annotation.",
            "Pag start sa recorder, keep a steady distance from the mic.",
            "Na postpone ang schedule, so reschedule na lang tayo tomorrow.",
            "I-check nato ang waveform kung naay clipping sa loud parts.",
            "Please read the prompt exactly, then add natural intonation.",
            "Sa chat kanina, we agreed to use the same naming convention.",
            "Kung okay ka, let's do one dry run before the final take.",
            "Niadto siya sa library and borrowed books about speech technology.",
            "Ayaw i-overthink, just relax and speak in your normal tone.",
            "I-compile nato ang results para ready sa advisor meeting.",
            "Nagpasalamat ang team kay complete ang submitted recordings.",
            "Paghuman sa batch, clean up the desk and shut down the PC.",
            "Naa koy correction sa transcript, please replace one missing word.",
            "Kung ready na tanan, start na ta sa validation stage.",
            "Sa weekend workshop, we practiced reading with consistent pacing.",
            "Paki-rename ang files according sa required pattern.",
            "Dili ko kaapil later, but I already uploaded my part.",
            "Please confirm kung kinsa ang next speaker sa lineup.",
            "Nag handa kami ng checklist para wala tayong makalimutan.",
            "Karong gabii, i-review ang consent notes and privacy reminders.",
            "Final na ni nga set, submit everything before midnight please."
        ]
        texts.extend(additional_texts)
        return texts
    
    def display_menu(self):
        """Display the main menu"""
        stats = self.dataset.get_dataset_stats()
        
        print("\n" + "="*60)
        print("VOICE CLONING AUDIO DATA RECORDER (CEBUANO)")
        print("="*60)
        print(f"Current Dataset: {self.dataset.dataset_dir}")
        print(f"Files: {stats['total_files']} | Duration: {stats['total_duration']:.1f}s")
        print(f"Average: {stats['average_duration']:.1f}s per file")
        print("-"*60)
        print("1. Record with suggested text")
        print("2. Record with custom text")
        print("3. Batch recording session")
        print("4. List audio devices")
        print("5. View dataset statistics")
        print("6. Export dataset")
        print("7. Quit")
        print("-"*60)
    
    def record_with_text(self, text: str, speaker_name: str = "default"):
        """Record audio with given text"""
        print(f"\nText to read:")
        print(f'"{text}"')
        print("\nPress Enter when ready to record, or 'skip' to skip this text:")
        
        user_input = input().strip().lower()
        if user_input == 'skip':
            return False
        
        print("\nRecording will start in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("🔴 RECORDING NOW - Start speaking!")
        
        # Record for a reasonable duration or until silence
        try:
            audio_data = self.recorder.record_audio(duration=10)  # 10 second max
            
            if len(audio_data) > 0:
                # Preview the recording
                print("\nRecording captured! Press Enter to save, 'r' to re-record, 's' to skip:")
                choice = input().strip().lower()
                
                if choice == 'r':
                    return self.record_with_text(text, speaker_name)
                elif choice == 's':
                    return False
                else:
                    # Save the recording
                    filename = self.dataset.add_recording(audio_data, text, speaker_name)
                    print(f"✅ Saved: {filename}")
                    return True
            else:
                print("No audio recorded. Try again.")
                return False
                
        except Exception as e:
            print(f"Recording error: {e}")
            return False
    
    def batch_recording_session(self, speaker_name: str = "default", num_samples: int = 20):
        """Record multiple samples in batch"""
        print(f"\n📝 BATCH RECORDING SESSION")
        print(f"Target: {num_samples} recordings")
        print(f"Speaker: {speaker_name}")
        
        successful_recordings = 0
        
        for i, text in enumerate(self.sample_texts[:num_samples]):
            print(f"\n--- Recording {i+1}/{num_samples} ---")
            
            if self.record_with_text(text, speaker_name):
                successful_recordings += 1
                print(f"✅ Progress: {successful_recordings}/{num_samples}")
            else:
                print("⏭️  Skipped")
            
            # Ask if user wants to continue
            if i < num_samples - 1:
                print("\nContinue? (Enter/y to continue, 'q' to quit batch):")
                if input().strip().lower() == 'q':
                    break
        
        print(f"\n🎉 Batch session complete!")
        print(f"Successfully recorded: {successful_recordings} samples")
    
    def run(self):
        """Run the interactive recording session"""
        print("Welcome to the Voice Cloning Data Recorder (Cebuano)!")
        
        # Get speaker name
        speaker_name = input("Enter speaker name (or press Enter for 'default'): ").strip()
        if not speaker_name:
            speaker_name = "default"
        
        while True:
            self.display_menu()
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                # Record with suggested text
                import random
                text = random.choice(self.sample_texts)
                self.record_with_text(text, speaker_name)
                
            elif choice == '2':
                # Record with custom text
                custom_text = input("Enter text to record: ").strip()
                if custom_text:
                    self.record_with_text(custom_text, speaker_name)
                
            elif choice == '3':
                # Batch recording
                try:
                    num_samples = int(input("How many samples to record? (default 20): ") or "20")
                    self.batch_recording_session(speaker_name, num_samples)
                except ValueError:
                    print("Invalid number, using default 20")
                    self.batch_recording_session(speaker_name, 20)
                
            elif choice == '4':
                # List audio devices
                self.recorder.list_audio_devices()
                
            elif choice == '5':
                # View statistics
                stats = self.dataset.get_dataset_stats()
                print(f"\n📊 Dataset Statistics:")
                print(f"Total files: {stats['total_files']}")
                print(f"Total duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)")
                print(f"Average duration: {stats['average_duration']:.1f} seconds per file")
                print(f"Speakers: {', '.join(stats['speakers'])}")
                
            elif choice == '6':
                # Export dataset info
                print(f"\n📁 Dataset Location: {self.dataset.dataset_dir.absolute()}")
                print(f"📄 Transcripts file: {self.dataset.transcript_file}")
                print(f"📋 Metadata file: {self.dataset.metadata_file}")
                print("\nTo use with voice cloning script:")
                print(f"python voice_clone.py --mode train --data-dir {self.dataset.dataset_dir} --transcript-file {self.dataset.transcript_file}")
                
            elif choice == '7':
                # Quit
                print("Goodbye! Your recordings are saved in:", self.dataset.dataset_dir)
                break
                
            else:
                print("Invalid choice. Please select 1-7.")

def main():
    parser = argparse.ArgumentParser(description="Audio Data Collection Tool for Voice Cloning (Cebuano)")
    parser.add_argument("--dataset-dir", type=str, default="audio_data", 
                       help="Directory to store audio dataset")
    parser.add_argument("--mode", choices=["interactive", "single", "batch"], default="interactive",
                       help="Recording mode")
    parser.add_argument("--text", type=str, help="Text to record (for single mode)")
    parser.add_argument("--speaker", type=str, default="default", help="Speaker name")
    parser.add_argument("--count", type=int, default=20, help="Number of recordings for batch mode")
    parser.add_argument(
        "--agree-policy",
        action="store_true",
        help="Acknowledge repository contributor policy terms non-interactively.",
    )
    
    args = parser.parse_args()

    if not require_contributor_acknowledgement(assume_yes=args.agree_policy):
        return
    
    if args.mode == "interactive":
        # Run interactive mode
        recorder = InteractiveRecorder(args.dataset_dir)
        recorder.run()
        
    elif args.mode == "single":
        # Single recording
        if not args.text:
            print("Error: --text required for single mode")
            return
        
        recorder = InteractiveRecorder(args.dataset_dir)
        recorder.record_with_text(args.text, args.speaker)
        
    elif args.mode == "batch":
        # Batch recording
        recorder = InteractiveRecorder(args.dataset_dir)
        recorder.batch_recording_session(args.speaker, args.count)

if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) == 1:
        print("Audio Data Collection Tool for Voice Cloning (Cebuano)")
        print("="*50)
        print("\nQuick Start (Interactive Mode):")
        print("python audio_data_creator_cebuano.py")
        print("\nOther Usage Examples:")
        print("python audio_data_creator_cebuano.py --mode single --text 'Maligayang umaga' --speaker maria")
        print("python audio_data_creator_cebuano.py --mode batch --count 30 --speaker juan")
        print("\nRequired packages:")
        print("pip install sounddevice soundfile pyaudio numpy")
        print("\nStarting interactive mode...")

        if not require_contributor_acknowledgement():
            sys.exit(0)
        
        # Start interactive mode by default
        recorder = InteractiveRecorder()
        recorder.run()
    else:
        main()
