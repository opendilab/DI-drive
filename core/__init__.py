__TITLE__ = "DI-drive"
__VERSION__ = "0.3.0"
__DESCRIPTION__ = "Decision AI Auto-Driving Platform"
__AUTHOR__ = "OpenDILab Contributors"
__AUTHOR_EMAIL__ = "opendilab.contact@gmail.com"
__version__ = __VERSION__

SIMULATORS = ['carla', 'metadrive']

if 'carla' in SIMULATORS:
    try:
        import carla
    except:
        raise ImportError("Import carla failed! Please install carla Python API first.")
if 'metadrive' in SIMULATORS:
    try:
        import metadrive
    except:
        raise ImportError("Import metadrive failed! Please install metadrive simulator first.")
