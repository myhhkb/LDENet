import subprocess

model_cfgs = [
    # "runs/train/Mamba-YOLO-B2/weights/last.pt"
    # "ultralytics/cfg/models/hyper-yolo/hyper-yolo.yaml",
    # "ultralytics/cfg/models/12/yolo12.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-CGLU.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-CGLU-DYT.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-DFFN.yaml",
    # "runs/train/yolo12-A2C2f-DFFN-DYT/weights/last.pt",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-DFFN-DYT-Mona.yaml",
    # "runs/train/yolo12-A2C2f-DYT/weights/last.pt",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-EDFFN.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-FMFFN.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-FMFFN-DYT.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-FRFN.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-KAN.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-Mona.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-SEFFN.yaml",
    # "ultralytics/cfg/models/12/yolo12-A2C2f-SEFN.yaml",
    # "ultralytics/cfg/models/12/yolo12-CTrans.yaml",
    # "ultralytics/cfg/models/12/yolo12-FCM.yaml",
    # "ultralytics/cfg/models/12/yolo12-MutilBackbone-DAF.yaml",
    # "ultralytics/cfg/models/12/yolo12-TADDH.yaml",

    # C3K2
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_AdditiveBlock.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_AKConv.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_AP.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_CAMixer.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_CFBlock.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_ContextGuided.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_ConvAttn.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_Converse.yaml",


    # DY
    # #11 "ultralytics/cfg/models/12/yolo12-dy/yolo12-C3k2_DynamicConv.yaml",
    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-C3k2_DynamicFilter.yaml",
    # #11 "ultralytics/cfg/models/12/yolo12-dy/yolo12-C3k2_DySnakeConv.yaml",
    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-Detect_DyHeadWithDCNV3.yaml",
    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-Detect_DyHeadWithDCNV4.yaml",
    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-dyhead.yaml",
    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-Dynamic_HGBlock.yaml",
    # # 11"ultralytics/cfg/models/12/yolo12-dy/yolo12-DySample.yaml",


    # Down
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-Detect_Conv2Formers.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-ContextGuidedBlock_Down.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-Detect_LQE.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-Detect_LSCD.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-EIEStem.yaml",
    # "runs/Down/yolo12-EUCB_SC/weights/last.pt",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-FourierConv.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-GCConv.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-GDSAFusion.yaml",
    # "runs/Down/yolo12-LDConv/weights/last.pt",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-LoGStem.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_ConvFormer.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DBB.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DBlock.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DCMB.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DCNv2_Dynamic.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DCNv4.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DEConv.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DeepDBB.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DRB.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DSAN.yaml",
    # "ultralytics/cfg/models/12/yolo12-C3K2/yolo12-C3k2_DWR.yaml",

    # "runs/Down/yolo12-C3k2_FCA_CTA/weights/last.pt",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-ADown.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-WFU.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-wConv2d.yaml",
    # "ultralytics/cfg/models/12/yolo12-down/yolo12-TransNeXt_AggregatedAttention.yaml",

    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-WaveletPool.yaml",
    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-V7DownSampling.yaml",
    # "ultralytics/cfg/models/12/yolo12-dy/yolo12-Detect_SEAM.yaml"
    # "ultralytics/cfg/models/11/yolo11.yaml",

    # new
    # # "ultralytics/cfg/models/12/new/yolo12.yaml",
    # # "runs/YOLO12/yolo12-C3k2-MutilScaleEdgeInformationEnhance_1/weights/last.pt",
    # "ultralytics/cfg/models/12/new/yolo12-C3k2-MutilScaleEdgeInformationEnhance_2.yaml",  # a2c2f维度不匹配
    # # "ultralytics/cfg/models/12/new/yolo12-C3k2-MutilScaleEdgeInformationSelect.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-C3k2-SMPCGLU_1.yaml",  # 预编译模块未加载
    # "ultralytics/cfg/models/12/new/yolo12-C3k2-SMPCGLU_2.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-CGRFPN.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-ContextGuideFPN.yaml",
    # # "ultralytics/cfg/models/12/new/yolo12-CSP-FreqSpatial.yaml",
    # # "ultralytics/cfg/models/12/new/yolo12-CSP-PTB_1.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-EMBSFPN.yaml",
    # # "ultralytics/cfg/models/12/new/yolo12-FeaturePyramidSharedConv.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-LSDECD.yaml",
    # # "ultralytics/cfg/models/12/new/yolo12-MutilBackbone-DAF_1.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-ReCalibrationFPN-P345.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-ReCalibrationFPN-P2345.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-ReCalibrationFPN-P3456.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-TADDH.yaml",
    # "ultralytics/cfg/models/12/new/yolo12_C3K2-DIMB.yaml",
    # "ultralytics/cfg/models/12/new/yolo12_GlobalEdgeInformationTransfer1.yaml",
    # "ultralytics/cfg/models/12/new/yolo12_GlobalEdgeInformationTransfer2.yaml",
    # "ultralytics/cfg/models/12/new/yolo12_GlobalEdgeInformationTransfer3.yaml",
    # "ultralytics/cfg/models/12/new/yolo12_HAFB-1.yaml",
    # "ultralytics/cfg/models/12/new/yolo12_HAFB-2.yaml",
    # # "ultralytics/cfg/models/12/new/yolo12_MutilBackbone-HAFB.yaml",
    # "ultralytics/cfg/models/12/new/yolo12_MutilBackbone-MSGA.yaml",
    # "ultralytics/cfg/models/12/new/yolo12-SOEP.yaml",

# new11
#     "ultralytics/cfg/models/11/yolo11.yaml",
#     "ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationEnhance.yaml",
#     # "ultralytics/cfg/models/12/new/yolo12-C3k2-MutilScaleEdgeInformationEnhance_2.yaml",  # a2c2f维度不匹配
#     "ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationSelect.yaml",
#     "ultralytics/cfg/models/11/yolo1-C3k2-SMPCGLU.yaml",  # 预编译模块未加载
#     # "ultralytics/cfg/models/11/yolo12-C3k2-SMPCGLU_2.yaml",
#     "ultralytics/cfg/models/11/yolo11-CGRFPN.yaml",
#     "ultralytics/cfg/models/11/yolo11-ContextGuideFPN.yaml",
#     "ultralytics/cfg/models/11/yolo11-CSP-FreqSpatial.yaml",
#     "ultralytics/cfg/models/11/yolo11-CSP-PTB.yaml",
    #"runs/YOLO11/yolo11-EMBSFPN/weights/last.pt",
    # "ultralytics/cfg/models/11/yolo11-FeaturePyramidSharedConv.yaml",
    # "ultralytics/cfg/models/11/yolo11-LSDECD.yaml",
    #"runs/YOLO11/yolo11-MutilBackbone-DAF/weights/last.pt",
    # "runs/YOLO11/yolo11-ReCalibrationFPN-P345/weights/last.pt",
    #"ultralytics/cfg/models/11/yolo11-ReCalibrationFPN-P2345.yaml",
    #"ultralytics/cfg/models/11/yolo11-ReCalibrationFPN-P3456.yaml",
    #"ultralytics/cfg/models/11/yolo11-TADDH.yaml",
    # "ultralytics/cfg/models/11/yolo11-C3K2-DIMB.yaml",
    # "ultralytics/cfg/models/11/yolo11-GlobalEdgeInformationTransfer1.yaml",
    # "ultralytics/cfg/models/11/yolo11-GlobalEdgeInformationTransfer2.yaml",
    # "ultralytics/cfg/models/11/yolo11-GlobalEdgeInformationTransfer3.yaml",
    # "ultralytics/cfg/models/11/yolo11-MutilBackbone-HAFB.yaml",
    # "ultralytics/cfg/models/11/yolo11-MutilBackbone-MSGA.yaml",
    # #"ultralytics/cfg/models/11/yolo11-SOEP.yaml",
    # "ultralytics/cfg/models/11/yolo11-HAFB-1.yaml",
    # "ultralytics/cfg/models/11/yolo12_HAFB-2.yaml",
    # "ultralytics/cfg/models/our/yolo11-C3k2-EIEM.yaml",
    # "ultralytics/cfg/models/our/yolo11-C3k2-EMSC.yaml",
    # "ultralytics/cfg/models/our/yolo11-C3k2-EMSCP.yaml",
    # "ultralytics/cfg/models/our/yolo11-C3k2-SMPCGLU.yaml",
    # "ultralytics/cfg/models/our/yolo11-CSP-PMSFA.yaml",
    # "ultralytics/cfg/models/our/yolo11-EIEStem.yaml",
    # "ultralytics/cfg/models/our/yolo11-FDPN.yaml",
    # "ultralytics/cfg/models/our/yolo11-FDPN-DASI.yaml",
    # "ultralytics/cfg/models/our/yolo11-HAFB-1.yaml",
    # "ultralytics/cfg/models/our/yolo11-HAFB-2.yaml",
    # "ultralytics/cfg/models/our/yolo11-LSCD.yaml",
    # "ultralytics/cfg/models/our/yolo11-LSCSBD.yaml",
    # "ultralytics/cfg/models/our/yolo11-RGCSPELAN.yaml",
    # "ultralytics/cfg/models/our/yolo11-EIEStem_LSDECD.yaml",
    # "ultralytics/cfg/models/our/yolo11-EIEStem_LSCD.yaml",

    #"ultralytics/cfg/models/our/yolo11-EIEStem_C3k2_DCMB.yaml",
    # "ultralytics/cfg/models/our/yolo11-EIEStem_C3k2_EMSCP.yaml",
    # "ultralytics/cfg/models/our/yolo11-EIEStem_CSP_PMSFA.yaml",
    # "ultralytics/cfg/models/hyper-yolo/hyper-yolo.yaml",
    #"ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml",

    # "runs/YOLO11/yolo11-GlobalEdgeInformationTransfer1-PMSFA/weights/last.pt",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-PMSFA_EIEM.yaml"
    # "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml"
    # "ultralytics/cfg/models/our/yolo11-MutilBackbone-DAF_PMSFA.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB.yaml",
    # "ultralytics/cfg/models/our/yolo11-CGRFPN_PMSFA.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-C3K2-DIMB-Detect_LSCD.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_SOEP.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_HAFB_1.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_EIEM.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_LAWDS.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_DYhead.yaml",

    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_C2ASSA.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_C2TSSA_DYT.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_conv2former.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_Detect_SEAM.yaml",
    # "ultralytics/cfg/models/our/yolo11-GlobalEdgeInformationTransfer1-DCMB_Detect_LSCD_LQE.yaml",
    # "ultralytics/cfg/models/our/yolo11s-GlobalEdgeInformationTransfer1-DCMB_Detect_SEAM.yaml",
    # "ultralytics/cfg/models/our/yolo11m-GlobalEdgeInformationTransfer1-DCMB_Detect_SEAM.yaml",
    # "ultralytics/cfg/models/our/yolo11l-GlobalEdgeInformationTransfer1-DCMB_Detect_SEAM.yaml",
    # "ultralytics/cfg/models/our/yolo11x-GlobalEdgeInformationTransfer1-DCMB_Detect_SEAM.yaml",
    # "runs/YOLO12/yolo12-GlobalEdgeInformationTransfer1-DCMB_Detect_SEAM/weights/last.pt"


    # 模块消融实验
    # "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_SAEHead.yaml",
    # "ultralytics/cfg/models/end/MODULES/yolo11_C3k2_DIM_FMIOU.yaml",
    # "ultralytics/cfg/models/end/MODULES/yolo11_FMIOU.yaml",
    # "ultralytics/cfg/models/end/MODULES/yolo11_GEIT_DIMB_FMIOU.yaml",


    # "runs/xiaorong_module/yolo11_SEAHead_FMIOU/weights/last.pt",
    # "ultralytics/cfg/models/end/MODULES/yolo11_GEIT_FMIOU.yaml",
    # "ultralytics/cfg/models/end/MODULES/yolo11_GEIT_SEAHead.yaml",
    # "runs/xiaorong_module/yolo11_SEAHead/weights/last.pt",
    # "runs/xiaorong_module/yolo13/weights/last.pt"

    # "ultralytics/cfg/models/end/MODULES/yolo11_GEIT_SEAHead_FMIOU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_MPDIoU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_InnerIoU.yaml"
    # "runs/xiaorong_loss/yolo11_GEIT_DIMB_SEAHead_GIoU/weights/last.pt"
    # # 需补充对比实验
    # # 损失函数对比实验
     # "runs/xiaorong_loss/yolo11_GEIT_DIMB_SEAHead_CIoU/weights/last.pt",
    # "runs/xiaorong_loss/yolo11_GEIT_DIMB_SEAHead_DIoU/weights/last.pt",
    # "runs/xiaorong_loss/yolo11_GEIT_DIMB_SEAHead_EIoU/weights/last.pt",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_FocalerIoU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_GIoU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_InnerIoU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_IOU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_MPDIoU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_ShapeIoU.yaml",
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_WIoU.yaml",
    # MAX VS AVG
    # "ultralytics/cfg/models/end/AVG_MAX/yolo11_GEIT_MAX.yaml",
    # "ultralytics/cfg/models/end/AVG_MAX/yolo11_GEIT_DIMB_SEAHead_FMIOU_MAX.yaml",
    # "runs/xiaorong_loss/yolo11_GEIT_DIMB_SEAHead_WIoU/weights/last.pt"
    # "ultralytics/cfg/models/end/LOSS/yolo11_GEIT_DIMB_SEAHead_FMIoU_0.85.yaml"
    # "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU_addchange.yaml"
    # "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU_laplacian.yaml"
    # "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU_prewitt.yaml"
    # "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU_canny_proxy.yaml"
    # "runs/xiaorong_loss/yolo11_DIMB_GEIT_SEAHead_FMIOU_prev_feat/weights/last.pt"
    # "runs/xiaorong_loss/yolo11_DIMB_GEIT_SEAHead_FMIOU_sobel2/weights/last.pt"
    # "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU_laplacian.yaml"

]

for cfg_path in model_cfgs:
    subprocess.run(["python", "train_single.py", cfg_path])
