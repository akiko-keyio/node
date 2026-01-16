

from nwm.ztd_nwm import ZTDNWMGenerator

def ztd_3dgrid(nwm_file,region):

    z=ZTDNWMGenerator(nwm_file,location=df,resample_h=(0,6000,100),interp_to_site=False,vertical_level="h")
    dfr=z.run()