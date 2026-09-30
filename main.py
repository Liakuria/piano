import os, uuid, shutil, subprocess, threading, traceback, json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE = Path(os.getenv('DATA_DIR', '/data')); BASE.mkdir(parents=True, exist_ok=True)
JOBS = {}; LOCK = threading.Lock()
ALLOWED = {'.mp4','.mov','.m4v','.webm','.mp3','.wav','.m4a','.aac','.flac','.ogg'}
INSTRUMENTS = {'piano','vocals','guitar','bass','drums'}
ARRANGEMENTS = {'faithful','easy','full'}
DIFFICULTIES = {'beginner','intermediate','advanced'}
STYLES = {'auto','pop','ballad','waltz','classical'}
app = FastAPI(title='Music to Piano API', version='6.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

class AnalyzeRequest(BaseModel):
    job_id: str
    instrument: str = Field(default='piano')
    arrangement: str = Field(default='full')
    hand_split: bool = True
    quantize: bool = True
    clean_notes: bool = True
    smart_piano: bool = True
    chord_symbols: bool = True
    accompaniment: str = 'auto'
    transpose: int = 0
    difficulty: str = 'intermediate'
    style: str = 'auto'
    preserve_melody: bool = True

def update(job, **kwargs):
    with LOCK: JOBS[job].update(kwargs)

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'Command failed: {" ".join(cmd)}\n{p.stdout[-6000:]}')
    return p.stdout

def ffmpeg_extract(src, wav):
    run(['ffmpeg','-y','-i',str(src),'-vn','-ac','2','-ar','44100','-c:a','pcm_s16le',str(wav)])

def normalize_audio(src, dst):
    run(['ffmpeg','-y','-i',str(src),'-af','highpass=f=35,lowpass=f=18000,loudnorm=I=-18:TP=-2:LRA=11','-ar','44100','-ac','2','-c:a','pcm_s16le',str(dst)])

def smart_harmonic(src, dst, instrument):
    import librosa, soundfile as sf
    y, sr = librosa.load(str(src), sr=44100, mono=True)
    if instrument in ('piano','guitar','vocals'):
        y, _ = librosa.effects.hpss(y, margin=(1.0, 2.0))
        y = librosa.util.normalize(y)
    elif instrument == 'bass':
        # Keep the fundamental-heavy low register while suppressing rumble.
        y = librosa.effects.preemphasis(y)
        y = librosa.util.normalize(y)
    sf.write(str(dst), y, sr)

def separate(audio, out):
    out.mkdir(parents=True, exist_ok=True)
    run(['python','-m','demucs','-n','htdemucs','--mp3','-o',str(out),str(audio)])
    model_dir = out/'htdemucs'/audio.stem
    stems = {p.stem:p for p in model_dir.glob('*.mp3')}
    if not stems: stems = {p.stem:p for p in model_dir.glob('*.wav')}
    return stems

def choose_stem(stems, instrument):
    if instrument in ('vocals','bass','drums'): return stems.get(instrument)
    return stems.get('other')

def transcribe(audio, midi):
    from basic_pitch.inference import predict_and_save
    outdir = midi.parent/'basic_pitch'; outdir.mkdir(exist_ok=True)
    predict_and_save([str(audio)], str(outdir), save_midi=True, sonify_midi=False, save_model_outputs=False, save_notes=False)
    candidates = sorted(outdir.glob('*.mid'), key=lambda p:p.stat().st_mtime, reverse=True)
    if not candidates: raise RuntimeError('Basic Pitch did not produce a MIDI file.')
    shutil.copy2(candidates[0], midi)

def drum_transcribe(audio, midi):
    import librosa
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    y,sr=librosa.load(str(audio),sr=22050,mono=True); env=librosa.onset.onset_strength(y=y,sr=sr)
    peaks=librosa.onset.onset_detect(onset_envelope=env,sr=sr,units='time',backtrack=True)
    tempo,_=librosa.beat.beat_track(y=y,sr=sr); tempo=float(tempo[0] if hasattr(tempo,'__len__') else tempo) if tempo else 120
    mid=MidiFile(ticks_per_beat=480); tr=MidiTrack(); mid.tracks.append(tr)
    tr.append(MetaMessage('set_tempo',tempo=int(60000000/max(tempo,40))))
    last=0
    for t in peaks:
        tick=max(0,int(float(t)*480*tempo/60)); delta=max(0,tick-last)
        frame=int(float(t)*sr); seg=y[max(0,frame-1024):min(len(y),frame+1024)]
        centroid=float(librosa.feature.spectral_centroid(y=seg,sr=sr).mean()) if len(seg) else 0
        rms=float(librosa.feature.rms(y=seg).mean()) if len(seg) else .05
        note=36 if centroid<1800 else 42 if centroid>6000 else 38
        vel=min(120,max(35,int(rms*900)))
        tr.append(Message('note_on',channel=9,note=note,velocity=vel,time=delta)); tr.append(Message('note_off',channel=9,note=note,velocity=0,time=60)); last=tick+60
    mid.save(str(midi))

def midi_notes(src):
    from mido import MidiFile
    mid=MidiFile(str(src)); notes=[]; tempo=500000
    for tr in mid.tracks:
        abs_tick=0; active={}
        for msg in tr:
            abs_tick += msg.time
            if msg.type=='set_tempo': tempo=msg.tempo
            if msg.type=='note_on' and msg.velocity>0: active[(msg.channel,msg.note)]=(abs_tick,msg.velocity)
            elif msg.type in ('note_off','note_on') and (msg.type=='note_off' or msg.velocity==0):
                k=(msg.channel,msg.note)
                if k in active:
                    s,v=active.pop(k); notes.append({'start':s,'end':max(s+1,abs_tick),'channel':msg.channel,'note':msg.note,'velocity':v})
    return mid.ticks_per_beat, tempo, notes

def clean_quantize(src,dst,quantize=True):
    from mido import MidiFile,MidiTrack,Message,MetaMessage
    ticks,tempo,notes=midi_notes(src); grid=max(1,ticks//4); seen={}
    for n in notes:
        s,e=n['start'],n['end']
        if quantize:
            s=int(round(s/grid)*grid); e=int(round(e/grid)*grid)
        if e<=s: e=s+grid
        e=min(e,s+ticks*16)
        if e-s < max(8,grid//3): continue
        k=(s,n['note'],n['channel'])
        if k not in seen or n['velocity']>seen[k]['velocity']: seen[k]={**n,'start':s,'end':e}
    cleaned=sorted(seen.values(),key=lambda n:(n['start'],n['note']))
    if not cleaned: raise RuntimeError('No musical notes were detected. Try a clearer recording or another instrument.')
    out=MidiFile(ticks_per_beat=ticks); tr=MidiTrack(); out.tracks.append(tr); tr.append(MetaMessage('set_tempo',tempo=tempo)); events=[]
    for n in cleaned:
        events.append((n['start'],1,Message('note_on',channel=0,note=n['note'],velocity=n['velocity'],time=0)))
        events.append((n['end'],0,Message('note_off',channel=0,note=n['note'],velocity=0,time=0)))
    last=0
    for t,o,m in sorted(events,key=lambda x:(x[0],x[1])): m.time=max(0,t-last); tr.append(m); last=t
    out.save(str(dst))

def arrange_for_piano(src, dst, source_instrument, mode):
    """Turn detected source notes into playable piano material.
    faithful = preserve pitch/duration; easy = reduce density/range; full = preserve notes and
    gently spread dense guitar/other chords across the piano range.
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    mid=MidiFile(str(src)); ticks,tempo,notes=midi_notes(src)
    if not notes: raise RuntimeError('No notes to arrange.')
    grouped={}
    for n in notes: grouped.setdefault(n['start'],[]).append(n)
    out=MidiFile(ticks_per_beat=ticks); tr=MidiTrack(); out.tracks.append(tr); tr.append(MetaMessage('set_tempo',tempo=tempo))
    arranged=[]
    for start, group in grouped.items():
        group=sorted(group,key=lambda x:x['note'])
        if source_instrument=='guitar':
            # Guitar is already chromatic; piano can reproduce the same pitches.
            # Easy mode keeps the strongest 3-4 notes and moves extreme notes toward piano range.
            if mode=='easy': group=sorted(group,key=lambda x:x['velocity'],reverse=True)[:4]; group=sorted(group,key=lambda x:x['note'])
            for n in group:
                pitch=max(36,min(96,n['note']))
                arranged.append({**n,'note':pitch,'channel':0})
        elif source_instrument=='bass':
            for n in group:
                pitch=max(28,min(67,n['note']))
                arranged.append({**n,'note':pitch,'channel':0})
        elif source_instrument=='vocals':
            # Melody stays in pitch; piano can play it in a comfortable register.
            for n in group:
                pitch=max(48,min(84,n['note']))
                arranged.append({**n,'note':pitch,'channel':0})
        elif source_instrument=='drums':
            # Map kick/snare/hat to a simple left/right-hand piano reduction.
            mapping={36:36,38:50,42:62}
            for n in group:
                arranged.append({**n,'note':mapping.get(n['note'],62),'channel':0})
        else:
            for n in group: arranged.append({**n,'channel':0})
    if mode=='easy':
        # Remove exact pitch duplicates at same onset and very fast ornamental clusters.
        unique={}
        for n in arranged:
            k=(n['start'],n['note']); unique[k]=n if k not in unique else max(unique[k],n,key=lambda x:x['velocity'])
        arranged=list(unique.values())
    events=[]
    for n in arranged:
        events.append((n['start'],1,Message('note_on',channel=0,note=n['note'],velocity=n['velocity'],time=0)))
        events.append((n['end'],0,Message('note_off',channel=0,note=n['note'],velocity=0,time=0)))
    last=0
    for t,o,m in sorted(events,key=lambda x:(x[0],x[1])): m.time=max(0,t-last); tr.append(m); last=t
    out.save(str(dst))

def split_hands(src,right,left):
    from mido import MidiFile,MidiTrack,Message,MetaMessage
    mid=MidiFile(str(src)); ticks,_,notes=midi_notes(src); pitches=[n['note'] for n in notes]
    if not pitches: raise RuntimeError('No notes for hand split.')
    # Search for a boundary that minimizes crossings while keeping both hands musically populated.
    best=(10**9,60)
    for boundary in range(48,73):
        low=sum(1 for p in pitches if p<boundary); high=len(pitches)-low
        if not low or not high: continue
        crossing=sum(1 for n in notes if (n['note']<boundary and n['note']>=boundary))
        balance=abs(low-high)
        cost=crossing*10+balance
        if cost<best[0]: best=(cost,boundary)
    boundary=best[1]
    def build(path,want_right):
        out=MidiFile(ticks_per_beat=mid.ticks_per_beat); tr=MidiTrack(); out.tracks.append(tr); events=[]
        for n in notes:
            if (n['note']>=boundary)==want_right:
                events.append((n['start'],1,Message('note_on',channel=0,note=n['note'],velocity=n['velocity'],time=0)))
                events.append((n['end'],0,Message('note_off',channel=0,note=n['note'],velocity=0,time=0)))
        last=0
        for t,o,m in sorted(events,key=lambda x:(x[0],x[1])): m.time=max(0,t-last); tr.append(m); last=t
        out.save(str(path))
    build(right,True); build(left,False); return boundary

def estimate_tempo_key(audio):
    import librosa, numpy as np
    y,sr=librosa.load(str(audio),sr=22050,mono=True)
    tempo,_=librosa.beat.beat_track(y=y,sr=sr); tempo=float(tempo[0] if hasattr(tempo,'__len__') else tempo) if tempo else 0
    chroma=librosa.feature.chroma_cqt(y=y,sr=sr).mean(axis=1); names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']; root=int(np.argmax(chroma))
    return round(tempo,1),names[root]

def add_chords(score,audio):
    import librosa, numpy as np
    from music21 import harmony
    y,sr=librosa.load(str(audio),sr=22050,mono=True); hop=512; chroma=librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop)
    names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']; templates=[]
    for r in range(12):
        for quality,ints in [('maj',[0,4,7]),('min',[0,3,7]),('7',[0,4,7,10])]:
            v=np.zeros(12); v[(r+np.array(ints))%12]=1; templates.append((r,quality,v))
    parts=list(score.parts)
    if not parts: return
    part=parts[0]; sec=max(1,int(sr*4/hop))
    for i in range(0,chroma.shape[1],sec):
        vec=chroma[:,i:i+sec].mean(axis=1); best=None
        for r,q,t in templates:
            sim=float(np.dot(vec,t)/(np.linalg.norm(vec)*np.linalg.norm(t)+1e-9))
            if best is None or sim>best[0]: best=(sim,r,q)
        if best and best[0]>.68:
            suffix={'min':'m','7':'7','maj':''}[best[2]]; cs=harmony.ChordSymbol(names[best[1]]+suffix); cs.quarterLength=4; part.insert((i/sec)*4,cs)


def detect_chord_progression(src):
    """Infer a compact chord progression directly from detected MIDI notes.
    Uses pitch-class templates and returns start/end in MIDI ticks.
    """
    from collections import defaultdict
    ticks, tempo, notes = midi_notes(src)
    if not notes: return []
    by_start=defaultdict(list)
    for n in notes: by_start[n['start']].append(n)
    onsets=sorted(by_start)
    names=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    templates=[]
    for root in range(12):
        for q,ints in [('maj',(0,4,7)),('min',(0,3,7)),('7',(0,4,7,10)),('sus4',(0,5,7))]:
            templates.append((root,q,{(root+i)%12 for i in ints}))
    out=[]
    for i,start in enumerate(onsets):
        end=onsets[i+1] if i+1<len(onsets) else start+ticks*2
        pcs=set(n['note']%12 for n in by_start[start])
        if len(pcs)<2: continue
        best=None
        for root,q,t in templates:
            overlap=len(pcs&t); extra=len(pcs-t)
            score=overlap*2-extra
            if best is None or score>best[0]: best=(score,root,q)
        if best and best[0]>=3:
            chord=f"{names[best[1]]}{'m' if best[2]=='min' else '7' if best[2]=='7' else 'sus4' if best[2]=='sus4' else ''}"
            if out and out[-1]['chord']==chord:
                out[-1]['end']=end
            else:
                out.append({'start':start,'end':end,'chord':chord,'root':best[1],'quality':best[2]})
    return out

def add_accompaniment(src, dst, mode='auto', arrangement='full', style='auto', difficulty='intermediate'):
    """Create a playable left-hand accompaniment from inferred chords.
    auto chooses block chords for sparse material and broken chords for dense guitar material.
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    mid=MidiFile(str(src)); ticks,tempo,notes=midi_notes(src)
    chords=detect_chord_progression(src)
    if not chords: shutil.copy2(src,dst); return []
    if mode=='auto':
        if style in ('ballad','waltz'): mode='broken'
        elif style in ('classical','arpeggio'): mode='arpeggio'
        else: mode='block' if len(notes)<30 else 'broken'
    if difficulty=='beginner' and mode=='arpeggio': mode='broken'
    out=MidiFile(ticks_per_beat=ticks); tr=MidiTrack(); out.tracks.append(tr); tr.append(MetaMessage('set_tempo',tempo=tempo))
    events=[]
    triads={'maj':[0,4,7],'min':[0,3,7],'7':[0,4,7,10],'sus4':[0,5,7]}
    for c in chords:
        root=36+c['root']
        ints=triads[c['quality']]
        pitches=[root+i for i in ints]
        if difficulty=='beginner':
            pitches=pitches[:3]
        elif difficulty=='advanced':
            pitches=pitches[:4]
        if style=='waltz':
            step=max(ticks,1); t=c['start']
            while t<c['end']:
                rootp=pitches[0]; fifth=pitches[min(2,len(pitches)-1)]
                en=min(c['end'],t+step//2)
                events += [(t,1,rootp),(en,0,rootp)]
                t2=min(c['end'],t+step//2)
                if t2<t+step:
                    events += [(t2,1,fifth),(min(c['end'],t2+step//2),0,fifth)]
                t+=step
        elif mode=='block':
            for p in pitches: events += [(c['start'],1,p),(c['end'],0,p)]
        else:
            step=max(ticks//2,1); t=c['start']; k=0
            order=list(range(len(pitches)))
            if mode=='arpeggio': order=order+order[::-1][1:-1]
            while t<c['end']:
                p=pitches[order[k%len(order)]]; en=min(c['end'],t+step)
                events += [(t,1,p),(en,0,p)]; t=en; k+=1
    last=0
    for t,on,p in sorted(events,key=lambda x:(x[0],x[1])):
        tr.append(Message('note_on' if on else 'note_off',channel=0,note=max(24,min(72,p)),velocity=72 if on else 0,time=max(0,t-last))); last=t
    out.save(str(dst))
    return chords

def apply_difficulty(src, dst, difficulty='intermediate', preserve_melody=True, instrument='piano'):
    """Shape note density/range for a human piano player."""
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    mid=MidiFile(str(src)); ticks,tempo,notes=midi_notes(src)
    if not notes: raise RuntimeError('No notes for difficulty shaping.')
    limits={'beginner':2,'intermediate':4,'advanced':8}
    max_sim=limits.get(difficulty,4)
    grouped={}
    for n in notes: grouped.setdefault(n['start'],[]).append(n)
    shaped=[]
    for start, group in grouped.items():
        group=sorted(group,key=lambda n:n['velocity'],reverse=True)
        if preserve_melody and instrument=='vocals' and group:
            keep=group[:max_sim]
        else:
            keep=group[:max_sim]
        if difficulty=='beginner':
            keep=sorted(keep,key=lambda n:n['note'])[-2:]
        shaped.extend(keep)
    out=MidiFile(ticks_per_beat=mid.ticks_per_beat); tr=MidiTrack(); out.tracks.append(tr); tr.append(MetaMessage('set_tempo',tempo=tempo))
    events=[]
    for n in shaped:
        events += [(n['start'],1,n),(n['end'],0,n)]
    last=0
    for t,on,n in sorted(events,key=lambda x:(x[0],x[1],x[2]['note'])):
        tr.append(Message('note_on' if on else 'note_off',channel=0,note=max(24,min(96,n['note'])),velocity=n['velocity'] if on else 0,time=max(0,t-last))); last=t
    out.save(str(dst))

def transpose_midi(src,dst,semitones):
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    mid=MidiFile(str(src)); out=MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for tr0 in mid.tracks:
        tr=MidiTrack(); out.tracks.append(tr)
        for msg in tr0:
            if msg.type in ('note_on','note_off'):
                d=msg.copy(note=max(0,min(127,msg.note+semitones)))
                tr.append(d)
            else: tr.append(msg.copy())
    out.save(str(dst))

def merge_midi_tracks(right_path,left_path,dst,tempo=500000):
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    out=MidiFile(ticks_per_beat=480); tr=MidiTrack(); out.tracks.append(tr); tr.append(MetaMessage('set_tempo',tempo=tempo))
    events=[]
    for path in (right_path,left_path):
        if not path or not path.exists(): continue
        _,_,notes=midi_notes(path)
        for n in notes:
            events += [(n['start'],1,n),(n['end'],0,n)]
    last=0
    for t,on,n in sorted(events,key=lambda x:(x[0],x[1],x[2]['note'])):
        tr.append(Message('note_on' if on else 'note_off',channel=0,note=n['note'],velocity=n['velocity'] if on else 0,time=max(0,t-last))); last=t
    out.save(str(dst))

def make_musicxml(midi,xml,right=None,left=None,audio=None,chords=True):
    from music21 import converter, stream
    score=stream.Score()
    if right and left and right.exists() and left.exists():
        rp=converter.parse(str(right)); lp=converter.parse(str(left)); rpart=rp.parts[0] if rp.parts else rp; lpart=lp.parts[0] if lp.parts else lp
        rpart.id='Right Hand'; lpart.id='Left Hand'; score.insert(0,rpart); score.insert(0,lpart)
    else: score=converter.parse(str(midi))
    if chords and audio: add_chords(score,audio)
    score.write('musicxml',fp=str(xml))

def make_pdf(xml,pdf):
    from music21 import converter
    score=converter.parse(str(xml)); ly=xml.with_suffix('.ly'); score.write('lily',fp=str(ly)); run(['lilypond','-o',str(pdf.with_suffix('')),str(ly)])
    if not pdf.with_suffix('.pdf').exists(): raise RuntimeError('LilyPond did not generate PDF.')

def worker(job,instrument,arrangement,do_hands,do_quantize,clean_notes,smart,chords,accompaniment,transpose,difficulty,style,preserve_melody):
    d=BASE/job
    try:
        src=Path(JOBS[job]['input_path']); update(job,status='processing',step='audio',progress=4,message='音声を抽出しています')
        wav=d/'audio.wav'; ffmpeg_extract(src,wav)
        norm=d/'normalized.wav'; update(job,step='preprocess',progress=11,message='スマホ録音を補正しています'); normalize_audio(wav,norm)
        update(job,step='separation',progress=24,message='AIで音源を分離しています'); stems=separate(norm,d/'separated')
        chosen=choose_stem(stems,instrument)
        if not chosen or not chosen.exists(): raise RuntimeError(f'No audio stem available for {instrument}.')
        target=d/'target.wav'; update(job,step='isolation',progress=36,message='目的楽器の音色を抽出・強調しています'); smart_harmonic(chosen,target,instrument) if smart else shutil.copy2(chosen,target)
        update(job,step='transcription',progress=49,message='音符・発音タイミングを解析しています'); raw=d/'raw.mid'; drum_transcribe(target,raw) if instrument=='drums' else transcribe(target,raw)
        cleaned=d/'clean.mid'; update(job,step='cleanup',progress=60,message='誤検出・重複・短すぎる音を整理しています'); clean_quantize(raw,cleaned,do_quantize)
        midi=d/'piano.mid'; update(job,step='arrangement',progress=68,message='ピアノで弾ける音域・密度へ変換しています'); arrange_for_piano(cleaned,midi,instrument,arrangement)
        midi2=d/'difficulty.mid'; apply_difficulty(midi,midi2,difficulty,preserve_melody,instrument); shutil.copy2(midi2,midi)
        if transpose:
            shifted=d/'transposed.mid'; transpose_midi(midi,shifted,transpose); midi=shifted; shutil.copy2(midi,d/'piano.mid'); midi=d/'piano.mid'
        chords_found=[]
        if instrument in ('guitar','piano','bass','vocals') and arrangement in ('full','easy'):
            update(job,step='chords',progress=72,message='コード進行を推定しています'); chords_found=detect_chord_progression(midi)
            if accompaniment in ('auto','block','broken','arpeggio') and instrument in ('guitar','piano') and arrangement=='full':
                update(job,step='accompaniment',progress=75,message='コードからピアノ伴奏を自動編曲しています')
                acc=d/'accompaniment.mid'; add_accompaniment(midi,acc,accompaniment,arrangement,style,difficulty)
                merged=d/'merged.mid'; merge_midi_tracks(midi,acc,merged); shutil.copy2(merged,midi)
        files={'midi':midi.name}
        boundary=None
        if do_hands and instrument in ('piano','guitar','vocals','bass'):
            update(job,step='hand_split',progress=79,message='左右の手を推定しています'); boundary=split_hands(midi,d/'right_hand.mid',d/'left_hand.mid'); files.update({'right_midi':'right_hand.mid','left_midi':'left_hand.mid'})
        tempo,key=estimate_tempo_key(norm); JOBS[job]['tempo']=tempo; JOBS[job]['key']=key; JOBS[job]['hand_boundary']=boundary; JOBS[job]['source_instrument']=instrument; JOBS[job]['arrangement']=arrangement
        JOBS[job]['chord_count']=len(chords_found)
        update(job,step='score',progress=87,message='コード・拍子を考慮して楽譜を生成しています')
        xml=d/'piano.musicxml'; make_musicxml(midi,xml,d/'right_hand.mid' if (d/'right_hand.mid').exists() else None,d/'left_hand.mid' if (d/'left_hand.mid').exists() else None,norm,chords); files['musicxml']=xml.name
        update(job,step='pdf',progress=94,message='PDF楽譜を組版しています'); pdf=d/'piano.pdf'; make_pdf(xml,pdf); files['pdf']=pdf.name
        update(job,status='done',step='complete',progress=100,message='ピアノ譜への変換が完了しました',files=files)
    except Exception as e:
        traceback.print_exc(); update(job,status='error',step='error',progress=100,error=str(e))

@app.get('/health')
def health(): return {'ok':True,'version':'6.0.0','features':['demucs','basic-pitch','instrument-to-piano','guitar-chord-reduction','chord-progression','auto-accompaniment','transpose','bass-left-hand','adaptive-hand-split','difficulty-aware-arrangement','style-aware-accompaniment','melody-preservation','chord-symbols','musicxml','lilypond']}

@app.post('/jobs')
async def create_job(file:UploadFile=File(...)):
    ext=Path(file.filename or '').suffix.lower()
    if ext not in ALLOWED: raise HTTPException(400,'Unsupported media format.')
    job=uuid.uuid4().hex; d=BASE/job; d.mkdir(parents=True); path=d/('input'+ext); total=0; max_bytes=int(os.getenv('MAX_UPLOAD_MB','300'))*1024*1024
    with open(path,'wb') as f:
        while chunk:=await file.read(1024*1024):
            total+=len(chunk)
            if total>max_bytes: shutil.rmtree(d,ignore_errors=True); raise HTTPException(413,'File too large.')
            f.write(chunk)
    JOBS[job]={'status':'uploaded','progress':0,'step':'waiting','message':'待機中','files':{},'input_path':str(path),'filename':file.filename}; return {'job_id':job}

@app.post('/jobs/{job_id}/analyze')
def analyze(job_id:str,req:AnalyzeRequest):
    if job_id not in JOBS: raise HTTPException(404,'Job not found')
    if req.job_id!=job_id or req.instrument not in INSTRUMENTS or req.arrangement not in ARRANGEMENTS or req.difficulty not in DIFFICULTIES or req.style not in STYLES: raise HTTPException(400,'Invalid request.')
    threading.Thread(target=worker,args=(job_id,req.instrument,req.arrangement,req.hand_split,req.quantize,req.clean_notes,req.smart_piano,req.chord_symbols,req.accompaniment,req.transpose,req.difficulty,req.style,req.preserve_melody),daemon=True).start(); return {'ok':True}

@app.get('/jobs/{job_id}')
def status(job_id:str):
    if job_id not in JOBS: raise HTTPException(404,'Job not found')
    return JOBS[job_id]

@app.get('/jobs/{job_id}/files/{name}')
def download(job_id:str,name:str):
    p=BASE/job_id/name
    if not p.exists() or p.parent!=BASE/job_id: raise HTTPException(404,'File not found')
    return FileResponse(p,filename=name)
