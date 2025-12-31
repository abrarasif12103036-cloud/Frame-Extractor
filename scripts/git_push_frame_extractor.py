import subprocess, sys, os
cwd = r"D:\frame-extractor"
print('cwd:', cwd)

def run(cmd):
    print('>',' '.join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print('ret', res.returncode)
    if res.stdout:
        print('out:', res.stdout.strip())
    if res.stderr:
        print('err:', res.stderr.strip())
    return res

# status
st = run(['git','status','--porcelain'])
if st.stdout.strip() == '':
    print('No local changes to commit.')
else:
    print('Changes present; staging and committing')
    run(['git','add','-A'])
    cm = run(['git','commit','-m','UI: remove color-name dropdown; use color picker only'])
    if cm.returncode != 0:
        print('Commit failed or nothing to commit')

# branch
br = run(['git','rev-parse','--abbrev-ref','HEAD'])
branch = br.stdout.strip() if br.returncode == 0 else None
if not branch:
    print('Could not determine branch; aborting push')
    sys.exit(1)

# push
print('Pushing to origin', branch)
push = run(['git','push','origin', branch])
if push.returncode != 0:
    # try set upstream
    print('Push failed; trying to set upstream and push')
    run(['git','push','-u','origin', branch])

print('Done')