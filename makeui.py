#!/Users/alex/anaconda/bin/python

import os, glob, subprocess

pycmd = 'python /Users/alex/anaconda/lib/python2.7/site-packages/PyQt4/uic/pyuic.py '

uifiles = glob.glob('openpiv/data/ui/*.ui')
for f in uifiles:
    path,fname = os.path.split(f)
    fname = 'ui_'+fname.rstrip('ui')+'py'
    path = 'openpiv/ui/'
    cmd = pycmd + f + ' -o ' + os.path.join(path,fname)
    print cmd
    subprocess.call(cmd)
    


pycmd = 'python /Users/alex/anaconda/lib/python2.7/site-packages/PyQt4/uic/pyrcc4.py '

rcfiles = glob.glob('openpiv/data/ui/*.qrc')
for f in rcfiles:
    path,fname = os.path.split(f)
    path = 'openpiv/ui/'
    fname = fname.rstrip('.qrc')+'_rc.py'
    cmd = pycmd + f + ' -o ' + os.path.join(path,fname)
    print cmd
    subprocess.call(cmd)

# pyrcc4 openpiv/data/ui_resources.qrc -o openpiv/ui/ui_resources_rc.py
