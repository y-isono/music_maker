import argparse
import pretty_midi
from synthesizer import Writer

def main(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    print(midi_data.instruments)
    audio_data = midi_data.synthesize()
    writer = Writer()
    writer.write_wave("./test.wav", audio_data)
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('midi_path', type=str)
    
    args = parser.parse_args()
    midi_path = args.midi_path
    
    main(midi_path)