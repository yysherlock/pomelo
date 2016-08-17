import os
import sys
import re

def txt2pdf(textfn, ps_prefix=None, pdf_prefix=None):
    """
    Convert a .txt file to .pdf file
    >>> txt2pdf('/home/luozhiyi/code/pyqt-ch04.txt')
    """
    text_prefix = os.path.splitext(textfn)[0]
    psfn = text_prefix+'.ps' if not ps_prefix else ps_prefix+'.ps'
    pdffn = text_prefix+'.pdf' if not pdf_prefix else pdf_prefix+'.pdf'
    try:
        os.system("enscript -p {0} {1}".format(psfn, textfn))
        os.system("ps2pdf {0} {1}".format(psfn, pdffn))
    except Exception as e:
        return "Failed, please check if you've installed `enscript` and `ps2pdf` commands:\n{}".format(e)

def copyTextFileTo(fn, outf, logfn=False):
    if logfn: outf.write('\n\n\n####################################\n'+fn+'\n####################################\n')
    with open(fn) as f:
        for line in f: outf.write(line)

def collectCodes(source_dir, outputfn,
                 filepats=[re.compile('\w+\.pyw?$'),re.compile('\w+\.(cpp|h|cc?)$'),
                           re.compile('\w+\.java$'),re.compile('\w+\.scala$')],
                 filterpats = [re.compile('.+\/build\/.+'), re.compile('.+\/UI[\/]?$'),
                               re.compile('.+\/youtub-dl[\/]?$'),
                               re.compile('.+\/demos?\/.+')]):
    """
    >>> collect(source_dir, outputfn)
    """
    if not os.path.exists(source_dir):
        raise ValueError("Not found {}".format(source_dir))
    with open(outputfn, 'w') as outf:
        for path, dirlist, filelist in os.walk(source_dir):
            for filename in filelist:
                if any([pat.match(path) for pat in filterpats]): continue
                if any([pat.match(filename) for pat in filepats]):
                    fn = os.path.join(path, filename)
                    copyTextFileTo(fn, outf, True)

def codes2pdf(source_dir, pdf_prefix):
    """
    collect a project source codes into a pdf file
    >>> from pomelo.tool import victorinox
    >>> victorinox.codes2pdf('MemN2N-tensorflow/','MemN2N')
    """
    textfn = pdf_prefix+'.txt'
    collectCodes(source_dir, textfn)
    txt2pdf(textfn)
