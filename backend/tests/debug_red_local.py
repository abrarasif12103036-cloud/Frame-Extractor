import subprocess, requests, zipfile, io
from PIL import Image

# generate
subprocess.run(['ffmpeg','-y','-f','lavfi','-i','color=c=black:s=160x120:d=1','-vf',"drawbox=x='t*100':y=50:w=8:h=8:color=red@1:t=fill",'debug_red.mp4'], check=True)
# upload
with open('debug_red.mp4','rb') as f:
    r = requests.post('http://127.0.0.1:5000/upload', files={'video':('debug_red.mp4', f, 'video/mp4')}, data={'interval':'0.1','track_red':'1'}, stream=True)
    print('status', r.status_code)
    if r.status_code!=200:
        try:
            print('json', r.json())
        except Exception as e:
            print('text', r.text)
    else:
        with open('debug_frames.zip','wb') as of:
            for chunk in r.iter_content(8192): of.write(chunk)
        with zipfile.ZipFile('debug_frames.zip') as z:
            names=[n for n in z.namelist() if n.lower().endswith('.jpg')]
            print('frames', len(names), names[:5])
            if names:
                data=z.read(names[0])
                img=Image.open(io.BytesIO(data)).convert('RGB')
                w,h=img.size
                print('size',w,h)
                pixels=img.load()
                # sample pixels across image
                for y in range(40,61):
                    row=[]
                    for x in range(0,w,10):
                        row.append(pixels[x,y])
                    print(y, row)
                # count green pixels
                total=0
                for py in range(h):
                    for px in range(w):
                        r,g,b=img.getpixel((px,py))
                        if g>150 and g>r+50 and g>b+50:
                            total+=1
                print('green count', total)
