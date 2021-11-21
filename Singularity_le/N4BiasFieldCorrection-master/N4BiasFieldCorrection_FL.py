import SimpleITK as sitk

def N4():
    print("N4 bias correction runs.")
    inputImage = sitk.ReadImage("Flair_reg_brain.nii.gz")
    # maskImage = sitk.ReadImage("06-t1c_mask.nii.gz")
    maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
    sitk.WriteImage(maskImage, "Flair_reg_brain_bias_mask.nii.gz")

    inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter();

    output = corrector.Execute(inputImage,maskImage)
    sitk.WriteImage(output,"Flair_reg_brain_bias.nii.gz")
    print("Finished N4 Bias Field Correction.....")

if __name__=='__main__':
   N4()
