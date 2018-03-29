import pip

def main():
    #install pydensecrf from https://github.com/lucasb-eyer/pydensecrf
    pip_install('git+https://github.com/lucasb-eyer/pydensecrf.git')

    #install py_gco from https://github.com/amueller/gco_python
    pip_install('git+git://github.com/amueller/gco_python')

def pip_install(pkg):
    pip.main(['install', pkg])

if __name__ == '__main__':
    main()
