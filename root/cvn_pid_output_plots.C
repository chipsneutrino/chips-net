void cvn_pid_output_plots(const char* outputFile = "../output/pid.txt",
                          const char* historyFile = "../output/pid_history.txt",
                          const char* plotFileName = "../plots/pid_plots.root"){

    gStyle->SetOptStat(0);

    // Fractions of the total event number for each event type expected in CHIPS
    float weightNueCCQEEvents       = 0.007;
    float weightNueCCnonQEEvents    = 0.028;
    float weightNumuCCQEEvents      = 0.133;
    float weightNumuCCnonQEEvents   = 0.532;
    float weightNCEvents            = 0.3;

    // Open and read the network output file into a outputTree
    TTree *outputTree = new TTree("outputTree", "outputTree");
    outputTree->ReadFile(outputFile, "label:beamE:p0:p1:p2:p3:p4:p5:p6:o0:o1:o2:o3:o4");


    outputTree->ReadFile(outputFile, "label:beamE:o0:o1:o2:o3:o4");

    // Open and read the network training file into a outputTree
    TTree *historyTree = new TTree("historyTree", "historyTree");
    historyTree->ReadFile(historyFile, "acc:val_acc:loss:val_loss");

    // Open the plot file to write output plots to
    TFile * plotFile = new TFile(plotFileName,"RECREATE");

    // Create all the histograms for the output file
    TH1F * hists[5][9];
    for (int eventType = 0; eventType<5; eventType++) {
        for(int outputType = 0; outputType<6; outputType++) {
            TString name = "hist_";
            name += eventType;
            name += "_";
            name += outputType;
            hists[eventType][outputType] = new TH1F(name, name, 50, 0.0, 2.0);
        }
    }

    // Calculate the weights we need to apply to each event type
    float w0 = (1.0/(float)outputTree->GetEntries("(label==0.0)"))*weightNueCCQEEvents * 1000;
    float w1 = (1.0/(float)outputTree->GetEntries("(label==1.0)"))*weightNueCCnonQEEvents * 1000;
    float w2 = (1.0/(float)outputTree->GetEntries("(label==2.0)"))*weightNumuCCQEEvents * 1000;
    float w3 = (1.0/(float)outputTree->GetEntries("(label==3.0)"))*weightNumuCCnonQEEvents * 1000;
    float w4 = (1.0/(float)outputTree->GetEntries("(label==4.0)"))*weightNCEvents * 1000;  

    // Calculate the weighted total events for each category
    float nuelCCQETot = (float)outputTree->GetEntries("(label==0.0)") * w0;
    float nuelCCnonQETot = (float)outputTree->GetEntries("(label==1.0)") * w1;
    float numuCCQETot = (float)outputTree->GetEntries("(label==2.0)") * w2;
    float numuCCnonQETot = (float)outputTree->GetEntries("(label==3.0)") * w3;
    float allNCTot = (float)outputTree->GetEntries("(label==4.0)") * w4;
        
    // Find where the maximum figure-of-merit lies...
    const int bins = 99;
    float eff_sig[bins], eff_bkg[bins], pur[bins], fom[bins], cut[bins];
    float maxFOM = 0.0; // Variable to keep track of maximum figure-of-merit
    float maxFomCut = 0;
    for (int bin=0; bin<bins; bin++) {
        cut[bin] = ((float)bin * 0.01) + 0.01;
                
        TString nuelCCQECutString = Form("(label==0.0)&&(o0+o1>%f)", cut[bin]);
        float nuelCCQECut = (float)outputTree->GetEntries(nuelCCQECutString.Data()) * w0;

        TString nuelCCnonQECutString = Form("(label==1.0)&&(o0+o1>%f)", cut[bin]);
        float nuelCCnonQECut = (float)outputTree->GetEntries(nuelCCnonQECutString.Data()) * w1;

        TString numuCCQECutString = Form("(label==2.0)&&(o0+o1>%f)", cut[bin]);
        float numuCCQECut = (float)outputTree->GetEntries(numuCCQECutString.Data()) * w2;

        TString numuCCnonQECutString = Form("(label==3.0)&&(o0+o1>%f)", cut[bin]);
        float numuCCnonQECut = (float)outputTree->GetEntries(numuCCnonQECutString.Data()) * w3;

        TString allNCCutString = Form("(label==4.0)&&(o0+o1>%f)", cut[bin]);
        float allNCCut = (float)outputTree->GetEntries(allNCCutString.Data()) * w4;

        eff_sig[bin] = ((nuelCCQECut+nuelCCnonQECut)/(nuelCCQETot+nuelCCnonQETot));
        eff_bkg[bin] = ((numuCCQECut+numuCCnonQECut+allNCCut)/(numuCCQETot+numuCCnonQETot+allNCTot));
        pur[bin] = ((nuelCCQECut+nuelCCnonQECut)/(nuelCCQECut+nuelCCnonQECut+numuCCQECut+numuCCnonQECut+allNCCut));
        fom[bin] = eff_sig[bin]*pur[bin];

        //std::cout << cut[bin] << ":" << eff_sig[bin] << "," << eff_bkg[bin] << "," << pur[bin] << "," << fom[bin] << std::endl;
      
        if (fom[bin]>maxFOM) { 
            maxFOM = fom[bin]; 
            maxFomCut = cut[bin];
        }
    }

    // Fill graphs and print maximum FOM to stdout
    TGraph *eff_sig_gr = new TGraph (bins, cut, eff_sig);
    eff_sig_gr->SetTitle("eff_sig_gr");
    eff_sig_gr->SetName("eff_sig_gr");
    TGraph *eff_bkg_gr = new TGraph (bins, cut, eff_bkg);
    eff_bkg_gr->SetTitle("eff_bkg_gr");
    eff_bkg_gr->SetName("eff_bkg_gr");
    TGraph *pur_gr = new TGraph (bins, cut, pur);
    pur_gr->SetTitle("pur_gr");
    pur_gr->SetName("pur_gr");
    TGraph *fom_gr = new TGraph (bins, cut, fom);
    fom_gr->SetTitle("fom_gr");
    fom_gr->SetName("fom_gr");
    std::cout << "Maximum Figure-of-merit: " << maxFOM << " at cut = " << maxFomCut << std::endl;

    // Fill histograms
    TString cat_0_string = Form("(%f*(label==0.0))", w0);
    outputTree->Draw("o0>>hist_0_0", cat_0_string.Data());
    outputTree->Draw("o1>>hist_0_1", cat_0_string.Data());
    outputTree->Draw("o2>>hist_0_2", cat_0_string.Data());
    outputTree->Draw("o3>>hist_0_3", cat_0_string.Data());
    outputTree->Draw("o4>>hist_0_4", cat_0_string.Data());
    outputTree->Draw("o0+o1>>hist_0_5", cat_0_string.Data());

    TString cat_1_string = Form("(%f*(label==1.0))", w1);
    outputTree->Draw("o0>>hist_1_0", cat_1_string.Data());
    outputTree->Draw("o1>>hist_1_1", cat_1_string.Data());
    outputTree->Draw("o2>>hist_1_2", cat_1_string.Data());
    outputTree->Draw("o3>>hist_1_3", cat_1_string.Data());
    outputTree->Draw("o4>>hist_1_4", cat_1_string.Data());
    outputTree->Draw("o0+o1>>hist_1_5", cat_1_string.Data());

    TString cat_2_string = Form("(%f*(label==2.0))", w2);
    outputTree->Draw("o0>>hist_2_0", cat_2_string.Data());
    outputTree->Draw("o1>>hist_2_1", cat_2_string.Data());
    outputTree->Draw("o2>>hist_2_2", cat_2_string.Data());
    outputTree->Draw("o3>>hist_2_3", cat_2_string.Data());
    outputTree->Draw("o4>>hist_2_4", cat_2_string.Data());
    outputTree->Draw("o0+o1>>hist_2_5", cat_2_string.Data());

    TString cat_3_string = Form("(%f*(label==3.0))", w3);
    outputTree->Draw("o0>>hist_3_0", cat_3_string.Data());
    outputTree->Draw("o1>>hist_3_1", cat_3_string.Data());
    outputTree->Draw("o2>>hist_3_2", cat_3_string.Data());
    outputTree->Draw("o3>>hist_3_3", cat_3_string.Data());
    outputTree->Draw("o4>>hist_3_4", cat_3_string.Data());
    outputTree->Draw("o0+o1>>hist_3_5", cat_3_string.Data());

    TString cat_4_string = Form("(%f*(label==4.0))", w4);
    outputTree->Draw("o0>>hist_4_0", cat_4_string.Data());
    outputTree->Draw("o1>>hist_4_1", cat_4_string.Data());
    outputTree->Draw("o2>>hist_4_2", cat_4_string.Data());
    outputTree->Draw("o3>>hist_4_3", cat_4_string.Data());
    outputTree->Draw("o4>>hist_4_4", cat_4_string.Data());
    outputTree->Draw("o0+o1>>hist_4_5", cat_4_string.Data());

    // Now I want to make the Efficiency and Purity Plot
    // First find all the efficiencies we need...
    TH1F* nueCCQEAll = new TH1F("nueCCQEAll", "", 8, 1000, 5000);
    TH1F* nueCCQESel = new TH1F("nueCCQESel", "", 8, 1000, 5000);
    nueCCQEAll->Sumw2();
    nueCCQESel->Sumw2();

    TH1F* nueCCnonQEAll = new TH1F("nueCCnonQEAll", "", 8, 1000, 5000);
    TH1F* nueCCnonQESel = new TH1F("nueCCnonQESel", "", 8, 1000, 5000);
    nueCCnonQEAll->Sumw2();
    nueCCnonQESel->Sumw2();

    TH1F* numuCCAll = new TH1F("numuCCAll", "", 8, 1000, 5000);
    TH1F* numuCCSel = new TH1F("numuCCSel", "", 8, 1000, 5000);
    numuCCAll->Sumw2();
    numuCCSel->Sumw2();

    TH1F* NCAll = new TH1F("NCAll", "", 8, 1000, 5000);
    TH1F* NCSel = new TH1F("NCSel", "", 8, 1000, 5000);
    NCAll->Sumw2();
    NCSel->Sumw2();

    TString nueCCQECutString = Form("(label==0.0)&&(o0+o1>%f)", maxFomCut);
    outputTree->Draw("beamE>>nueCCQEAll", "(label==0.0)");
    outputTree->Draw("beamE>>nueCCQESel", nueCCQECutString.Data());
    TEfficiency* nueCCQEEff = new TEfficiency(*nueCCQESel,*nueCCQEAll);

    TString nueCCnonQECutString = Form("(label==1.0)&&(o0+o1>%f)", maxFomCut);
    outputTree->Draw("beamE>>nueCCnonQEAll", "(label==1.0)");
    outputTree->Draw("beamE>>nueCCnonQESel", nueCCnonQECutString.Data());
    TEfficiency* nueCCnonQEEff = new TEfficiency(*nueCCnonQESel,*nueCCnonQEAll);

    TString numuCCQECutString = Form("(label==2.0)&&(o0+o1>%f)", maxFomCut);
    TString numuCCnonQECutString = Form("(label==3.0)&&(o0+o1>%f)", maxFomCut);
    outputTree->Draw("beamE>>numuCCAll", "(label==2.0)");
    outputTree->Draw("beamE>>numuCCAll", "(label==3.0)");
    outputTree->Draw("beamE>>numuCCSel", numuCCQECutString.Data());
    outputTree->Draw("beamE>>numuCCSel", numuCCnonQECutString.Data());
    TEfficiency* numuCCEff = new TEfficiency(*numuCCSel,*numuCCAll);

    TString allNCCutString = Form("(label==4.0)&&(o0+o1>%f)", maxFomCut);
    outputTree->Draw("beamE>>NCAll", "(label==4.0)");
    outputTree->Draw("beamE>>NCSel", allNCCutString.Data());
    TEfficiency* NCEff = new TEfficiency(*NCSel,*NCAll);

    // Now find the signal purity
    TH1F* nueCCSignal = new TH1F("nueCCSignal", "", 8, 1000, 5000);
    TH1F* nueCCTotal = new TH1F("nueCCTotal", "", 8, 1000, 5000);
    nueCCSignal->Sumw2();
    nueCCTotal->Sumw2();

    TString signal_string = Form("(%f*((label==0.0)&&(o0+o1>%f))) + (%f*((label==1.0)&&(o0+o1>%f)))", w0, maxFomCut, w1, maxFomCut);
    TString total_string  = Form("(%f*((label==0.0)&&(o0+o1>%f))) + (%f*((label==1.0)&&(o0+o1>%f))) + (%f*((label==2.0)&&(o0+o1>%f))) + (%f*((label==3.0)&&(o0+o1>%f))) + (%f*((label==4.0)&&(o0+o1>%f)))", w0, maxFomCut, w1, maxFomCut, w2, maxFomCut, w3, maxFomCut, w4, maxFomCut);
    outputTree->Draw("beamE>>nueCCSignal", signal_string.Data());
    outputTree->Draw("beamE>>nueCCTotal", total_string.Data());
    nueCCSignal->Divide(nueCCTotal);

    TCanvas *c5 = new TCanvas("c5", "", 800, 600);
    c5->cd();

    TH2F *hempty = new TH2F("hempty", ";Neutrino Energy (MeV); Efficiency or Purity", 1, 1000, 5000, 10, 0, 1);
    hempty->GetXaxis()->SetTitleSize(0.06);    hempty->GetYaxis()->SetTitleSize(0.06); hempty->GetXaxis()->CenterTitle();
    hempty->GetXaxis()->SetTitleOffset(0.8);   hempty->GetYaxis()->SetTitleOffset(0.8); hempty->GetYaxis()->CenterTitle();
    hempty->GetXaxis()->SetLabelSize(0.05);    hempty->GetYaxis()->SetLabelSize(0.05);
    hempty->Draw();

    nueCCQEEff->SetLineColor(kGreen);       nueCCQEEff->SetLineWidth(2);   nueCCQEEff->SetMarkerSize(1.2);    nueCCQEEff->SetMarkerStyle(20);    nueCCQEEff->SetMarkerColor(kGreen);    nueCCQEEff->Draw("sameP");
    nueCCnonQEEff->SetLineColor(kBlue);          nueCCnonQEEff->SetLineWidth(2);     nueCCnonQEEff->SetMarkerSize(1.2);      nueCCnonQEEff->SetMarkerStyle(20);      nueCCnonQEEff->SetMarkerColor(kBlue);       nueCCnonQEEff->Draw("sameP");
    numuCCEff->SetLineColor(kMagenta);    numuCCEff->SetLineWidth(2);    numuCCEff->SetMarkerSize(1.2);     numuCCEff->SetMarkerStyle(20);     numuCCEff->SetMarkerColor(kMagenta);   numuCCEff->Draw("sameP");
    NCEff->SetLineColor(kRed);              NCEff->SetLineWidth(2);        NCEff->SetMarkerSize(1.2);         NCEff->SetMarkerStyle(20);         NCEff->SetMarkerColor(kRed);       NCEff->Draw("sameP");
    nueCCSignal->SetLineColor(kBlack);      nueCCSignal->SetLineWidth(2);  nueCCSignal->SetMarkerSize(1.2);   nueCCSignal->SetMarkerStyle(20);   nueCCSignal->SetMarkerColor(kBlack);     nueCCSignal->Draw("sameP");

    TLegend *leg = new TLegend(0.7, 0.60, 0.9, 0.80, "Efficiency");
    leg->AddEntry(nueCCQEEff, "#nu_{e} CC QE", "P");
    leg->AddEntry(nueCCnonQEEff, "#nu_{e} CC nQE", "P");
    leg->AddEntry(numuCCEff, "#nu_{#mu} CC", "P");
    leg->AddEntry(NCEff, "NC", "P");
    leg->SetTextSize(0.03);
    leg->SetTextFont(42);
    leg->SetFillColor(42);
    leg->SetFillStyle(1001);
    leg->Draw();

    TLegend *leg2 = new TLegend(0.7, 0.52, 0.9, 0.60, "Purity");
    leg2->AddEntry(nueCCSignal, "#nu_{e} CC", "P");
    leg2->SetTextFont(42);
    leg2->SetFillColor(42);
    leg2->SetFillStyle(1001);
    leg2->SetTextSize(0.03);
    leg2->Draw();

    c5->SetGridy();
    c5->Update();
    c5->SaveAs("plots/cvn_pid_effPur.png");
    c5->SaveAs("plots/cvn_pid_effPur.C");

    ifstream history;
    history.open(historyFile);
    const int epochs = 21;
    float epoch_arr[epochs], acc_array[epochs], val_acc_array[epochs], loss_arr[epochs], val_loss_arr[epochs];
    if (history.is_open()) {
        for (int epoch = 0; epoch<epochs; epoch++) {
            epoch_arr[epoch] = epoch;
            if (epoch == 0) {
                acc_array[epoch] = 0.2;
                val_acc_array[epoch] = 0.2;
                loss_arr[epoch] = 9.99;
                val_loss_arr[epoch] = 9.99;
            } else {
                history >> acc_array[epoch];
                history >> val_acc_array[epoch];
                history >> loss_arr[epoch];
                history >> val_loss_arr[epoch];
            }
        }
    }
    history.close();

    TGraph *acc_gr = new TGraph (epochs, epoch_arr, acc_array);
    acc_gr->SetTitle("acc_gr");
    acc_gr->SetName("acc_gr");
    TGraph *val_acc_gr = new TGraph (epochs, epoch_arr, val_acc_array);
    val_acc_gr->SetTitle("val_acc_gr");
    val_acc_gr->SetName("val_acc_gr");
    TGraph *loss_gr = new TGraph (epochs, epoch_arr, loss_arr);
    loss_gr->SetTitle("loss_gr");
    loss_gr->SetName("loss_gr");
    TGraph *val_loss_gr = new TGraph (epochs, epoch_arr, val_loss_arr);
    val_loss_gr->SetTitle("val_loss_gr");
    val_loss_gr->SetName("val_loss_gr");

    // Write plots to file
    nueCCQEEff->Write();
    nueCCnonQEEff->Write();
    numuCCEff->Write();
    NCEff->Write();
    nueCCSignal->Write();
    eff_sig_gr->Write();
    eff_bkg_gr->Write();
    pur_gr->Write();
    fom_gr->Write();
    acc_gr->Write();
    val_acc_gr->Write();
    loss_gr->Write();
    val_loss_gr->Write();
    for (int eventType = 0; eventType<5; eventType++) {
        for(int outputType = 0; outputType<6; outputType++) {
            hists[eventType][outputType]->Write();
        }
    }

    plotFile->Close();

}
