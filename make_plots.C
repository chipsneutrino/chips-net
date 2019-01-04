void make_plots(){
    gStyle->SetOptStat(0);

    const char* outputFile = "/unix/chips/jtingey/CHIPS/data/CVN/plots/output/5_model2.txt";

    float weightNueCCQEEvents       = 0.007;
    float weightNueCCnonQEEvents    = 0.028;
    float weightNumuCCQEEvents      = 0.133;
    float weightNumuCCnonQEEvents   = 0.532;
    float weightNCEvents            = 0.3;
    std::cout << "Weights: (" << weightNueCCQEEvents << "," << weightNueCCnonQEEvents << "," << weightNumuCCQEEvents << "," << weightNumuCCnonQEEvents << "," << weightNCEvents << ")" << std::endl;
    std::cout << "Weight Sum = " << weightNueCCQEEvents+weightNueCCnonQEEvents+weightNumuCCQEEvents+weightNumuCCnonQEEvents+weightNCEvents << std::endl;

    float weightNueCC = 0.035;
    float weightNumuCC = 0.665;

    TTree *tree = new TTree("tree", "tree");
    tree->ReadFile(outputFile, "label:energy:o0:o1:o2:o3:o4");
    //tree->ReadFile(outputFile, "label:energy:o0:o1:o2");

    TFile * mainOutput = new TFile("5_model2.root","RECREATE");

    TH1F * hists[5][9];
    for (int eventType = 0; eventType<5; eventType++) {
        for(int outputType = 0; outputType<9; outputType++) {
            TString name = "hist_";
            name += eventType;
            name += "_";
            name += outputType;
            hists[eventType][outputType] = new TH1F(name, name, 50, 0.0, 1.0);
        }
    }

    float w0 = (1.0/(float)tree->GetEntries("(label==0.0)"))*weightNueCCQEEvents;
    float w1 = (1.0/(float)tree->GetEntries("(label==1.0)"))*weightNueCCnonQEEvents;
    float w2 = (1.0/(float)tree->GetEntries("(label==2.0)"))*weightNumuCCQEEvents;
    float w3 = (1.0/(float)tree->GetEntries("(label==3.0)"))*weightNumuCCnonQEEvents;
    float w4 = (1.0/(float)tree->GetEntries("(label==4.0)"))*weightNCEvents;  

    /*
    tree->Draw("o0>>hist_0_0", "label==0.0");
    tree->Draw("o1>>hist_0_1", "label==0.0");
    tree->Draw("o2>>hist_0_2", "label==0.0");

    tree->Draw("o0>>hist_1_0", "label==1.0");
    tree->Draw("o1>>hist_1_1", "label==1.0");
    tree->Draw("o2>>hist_1_2", "label==1.0");

    tree->Draw("o0>>hist_2_0", "label==2.0");
    tree->Draw("o1>>hist_2_1", "label==2.0");
    tree->Draw("o2>>hist_2_2", "label==2.0");
    */

    tree->Draw("o0>>hist_0_0", "0.007*(label==0.0)");
    tree->Draw("o1>>hist_0_1", "0.007*(label==0.0)");
    tree->Draw("o2>>hist_0_2", "0.007*(label==0.0)");
    tree->Draw("o3>>hist_0_3", "0.007*(label==0.0)");
    tree->Draw("o4>>hist_0_4", "0.007*(label==0.0)");
    tree->Draw("o0+o1>>hist_0_5", "0.007*(label==0.0)");
    tree->Draw("o2+o3>>hist_0_6", "0.007*(label==0.0)");
    tree->Draw("o0+o1-o2-o3-o4>>hist_0_7", "0.007*(label==0.0)");
    tree->Draw("o0+o1+o2+o3+o4>>hist_0_8", "0.007*(label==0.0)");

    tree->Draw("o0>>hist_1_0", "0.028*(label==1.0)");
    tree->Draw("o1>>hist_1_1", "0.028*(label==1.0)");
    tree->Draw("o2>>hist_1_2", "0.028*(label==1.0)");
    tree->Draw("o3>>hist_1_3", "0.028*(label==1.0)");
    tree->Draw("o4>>hist_1_4", "0.028*(label==1.0)");
    tree->Draw("o0+o1>>hist_1_5", "0.028*(label==1.0)");
    tree->Draw("o2+o3>>hist_1_6", "0.028*(label==1.0)");
    tree->Draw("o0+o1-o2-o3-o4>>hist_1_7", "0.028*(label==1.0)");
    tree->Draw("o0+o1+o2+o3+o4>>hist_1_8", "0.028*(label==1.0)");

    tree->Draw("o0>>hist_2_0", "0.133*(label==2.0)");
    tree->Draw("o1>>hist_2_1", "0.133*(label==2.0)");
    tree->Draw("o2>>hist_2_2", "0.133*(label==2.0)");
    tree->Draw("o3>>hist_2_3", "0.133*(label==2.0)");
    tree->Draw("o4>>hist_2_4", "0.133*(label==2.0)");
    tree->Draw("o0+o1>>hist_2_5", "0.133*(label==2.0)");
    tree->Draw("o2+o3>>hist_2_6", "0.133*(label==2.0)");
    tree->Draw("o0+o1-o2-o3-o4>>hist_2_7", "0.133*(label==2.0)");
    tree->Draw("o0+o1+o2+o3+o4>>hist_2_8", "0.133*(label==2.0)");

    tree->Draw("o0>>hist_3_0", "0.532*(label==3.0)");
    tree->Draw("o1>>hist_3_1", "0.532*(label==3.0)");
    tree->Draw("o2>>hist_3_2", "0.532*(label==3.0)");
    tree->Draw("o3>>hist_3_3", "0.532*(label==3.0)");
    tree->Draw("o4>>hist_3_4", "0.532*(label==3.0)");
    tree->Draw("o0+o1>>hist_3_5", "0.532*(label==3.0)");
    tree->Draw("o2+o3>>hist_3_6", "0.532*(label==3.0)");
    tree->Draw("o0+o1-o2-o3-o4>>hist_3_7", "0.532*(label==3.0)");
    tree->Draw("o0+o1+o2+o3+o4>>hist_3_8", "0.532*(label==3.0)");

    tree->Draw("o0>>hist_4_0", "0.3*(label==4.0)");
    tree->Draw("o1>>hist_4_1", "0.3*(label==4.0)");
    tree->Draw("o2>>hist_4_2", "0.3*(label==4.0)");
    tree->Draw("o3>>hist_4_3", "0.3*(label==4.0)");
    tree->Draw("o4>>hist_4_4", "0.3*(label==4.0)");
    tree->Draw("o0+o1>>hist_4_5", "0.3*(label==4.0)");
    tree->Draw("o2+o3>>hist_4_6", "0.3*(label==4.0)");
    tree->Draw("o0+o1-o2-o3-o4>>hist_4_7", "0.3*(label==4.0)");
    tree->Draw("o0+o1+o2+o3+o4>>hist_4_8", "0.3*(label==4.0)");

    // Now I want to make the Efficiency and Purity Plot
    // First find all the efficiencies we want...

    TH1F* nueCCQEAll = new TH1F("nueCCQEAll", "", 8, 1000, 5000);
    TH1F* nueCCQESel = new TH1F("nueCCQESel", "", 8, 1000, 5000);
    nueCCQEAll->Sumw2();
    nueCCQESel->Sumw2();

    TH1F* nueCCAll = new TH1F("nueCCnonQEAll", "", 8, 1000, 5000);
    TH1F* nueCCSel = new TH1F("nueCCnonQESel", "", 8, 1000, 5000);
    nueCCAll->Sumw2();
    nueCCSel->Sumw2();

    TH1F* numuCCQEAll = new TH1F("numuCCQEAll", "", 8, 1000, 5000);
    TH1F* numuCCQESel = new TH1F("numuCCQESel", "", 8, 1000, 5000);
    numuCCQEAll->Sumw2();
    numuCCQESel->Sumw2();

    TH1F* numuCCAll = new TH1F("numuCCnonQEAll", "", 8, 1000, 5000);
    TH1F* numuCCSel = new TH1F("numuCCnonQESel", "", 8, 1000, 5000);
    numuCCAll->Sumw2();
    numuCCSel->Sumw2();

    TH1F* NCAll = new TH1F("NCAll", "", 8, 1000, 5000);
    TH1F* NCSel = new TH1F("NCSel", "", 8, 1000, 5000);
    NCAll->Sumw2();
    NCSel->Sumw2();

    tree->Draw("energy>>nueCCQEAll", "(label==0.0)");
    tree->Draw("energy>>nueCCQESel", "(label==0.0) && (o0+o1>0.9)");
    TEfficiency* nueCCQEEff = new TEfficiency(*nueCCQESel,*nueCCQEAll);

    tree->Draw("energy>>nueCCnonQEAll", "(label==1.0)");
    tree->Draw("energy>>nueCCnonQESel", "(label==1.0) && (o0+o1>0.9)");
    TEfficiency* nueCCEff = new TEfficiency(*nueCCnonQESel,*nueCCnonQEAll);

    tree->Draw("energy>>numuCCQEAll", "(label==2.0)");
    tree->Draw("energy>>numuCCQESel", "(label==2.0) && (o0+o1>0.9)");
    TEfficiency* numuCCQEEff = new TEfficiency(*numuCCQESel,*numuCCQEAll);

    tree->Draw("energy>>numuCCnonQEAll", "(label==3.0)");
    tree->Draw("energy>>numuCCnonQESel", "(label==3.0) && (o0+o1>0.9)");
    TEfficiency* numuCCEff = new TEfficiency(*numuCCnonQESel,*numuCCnonQEAll);

    tree->Draw("energy>>NCAll", "(label==4.0)");
    tree->Draw("energy>>NCSel", "(label==4.0) && (o0+o1>0.9)");
    TEfficiency* NCEff = new TEfficiency(*NCSel,*NCAll);

    // Now find the signal purity
    TH1F* nueCCSignal = new TH1F("nueCCSignal", "", 8, 1000, 5000);
    nueCCSignal->Sumw2();
    TH1F* nueCCTotal = new TH1F("nueCCTotal", "", 8, 1000, 5000);
    nueCCTotal->Sumw2();

    std::cout << w0 << "," << w1 << "," << w2 << "," << w3 << "," << w4 << std::endl;
    TString signal_string = Form("(%f*((label==0.0)&&(o0+o1>0.9))) + (%f*((label==1.0)&&(o0+o1>0.9)))", w0, w1);
    TString total_string  = Form("(%f*((label==0.0)&&(o0+o1>0.9))) + (%f*((label==1.0)&&(o0+o1>0.9))) + (%f*((label==2.0)&&(o0+o1>0.9))) + (%f*((label==3.0)&&(o0+o1>0.9))) + (%f*((label==4.0)&&(o0+o1>0.9)))", w0, w1, w2, w3, w4);
    tree->Draw("energy>>nueCCSignal", signal_string.Data());
    tree->Draw("energy>>nueCCTotal", total_string.Data());
    nueCCSignal->Divide(nueCCTotal);

    TCanvas *c5 = new TCanvas("c5", "", 800, 600);
    c5->cd();

    TH2F *hempty = new TH2F("hempty", ";Neutrino Energy (MeV); Efficiency or Purity", 8, 1000, 5000, 10, 0, 1);
    hempty->GetXaxis()->SetTitleSize(0.06);    hempty->GetYaxis()->SetTitleSize(0.06); hempty->GetXaxis()->CenterTitle();
    hempty->GetXaxis()->SetTitleOffset(0.8);   hempty->GetYaxis()->SetTitleOffset(0.8); hempty->GetYaxis()->CenterTitle();
    hempty->GetXaxis()->SetLabelSize(0.05);    hempty->GetYaxis()->SetLabelSize(0.05);
    hempty->Draw();

    nueCCQEEff->SetLineColor(kBlack);       nueCCQEEff->SetLineWidth(2);   nueCCQEEff->SetMarkerSize(1.2);    nueCCQEEff->SetMarkerStyle(20);    nueCCQEEff->SetMarkerColor(kBlack);    nueCCQEEff->Draw("sameP");
    nueCCEff->SetLineColor(kBlue);          nueCCEff->SetLineWidth(2);     nueCCEff->SetMarkerSize(1.2);      nueCCEff->SetMarkerStyle(20);      nueCCEff->SetMarkerColor(kBlue);       nueCCEff->Draw("sameP");
    numuCCQEEff->SetLineColor(kGreen+2);    numuCCQEEff->SetLineWidth(2);  numuCCQEEff->SetMarkerSize(1.2);   numuCCQEEff->SetMarkerStyle(20);   numuCCQEEff->SetMarkerColor(kGreen+2); numuCCQEEff->Draw("sameP");
    //numuCCEff->SetLineColor(kBlue+1);      numuCCEff->SetLineWidth(2);    numuCCEff->SetMarkerSize(1.2);     numuCCEff->SetMarkerStyle(20);     numuCCEff->SetMarkerColor(kBlue+1);   numuCCEff->Draw("sameP");
    NCEff->SetLineColor(kMagenta);          NCEff->SetLineWidth(2);        NCEff->SetMarkerSize(1.2);         NCEff->SetMarkerStyle(20);         NCEff->SetMarkerColor(kMagenta);       NCEff->Draw("sameP");
    nueCCSignal->SetLineColor(kRed);        nueCCSignal->SetLineWidth(2);  nueCCSignal->SetMarkerSize(1.2);   nueCCSignal->SetMarkerStyle(20);   nueCCSignal->SetMarkerColor(kRed);     nueCCSignal->Draw("sameP");

    /*
    TLegend *leg = new TLegend(0.12, 0.70, 0.32, 0.9, "Efficiency");
    leg->AddEntry(nueCCQEEff, "#nu_{e} CCQE", "P");
    leg->AddEntry(nueCCEff, "#nu_{e} CC", "P");
    //leg->AddEntry(numuCCEff, "#nu_{#mu} CC nonQE", "P");
    leg->AddEntry(numuCCQEEff, "#nu_{#mu} CC", "P");
    leg->AddEntry(NCEff, "NC", "P");
    leg->SetTextSize(0.03);
    leg->SetTextFont(42);
    leg->SetFillColor(0);
    leg->SetFillStyle(0);
    leg->Draw();

    TLegend *leg2 = new TLegend(0.12, 0.62, 0.32, 0.7, "Purity");
    leg2->AddEntry(nueCCSignal, "#nu_{e} CC", "P");
    leg2->SetTextFont(42);
    leg2->SetFillColor(0);
    leg2->SetFillStyle(0);
    leg2->SetTextSize(0.03);
    leg2->Draw();
    */

    c5->Update();
    c5->SaveAs("effPur.png");
    //c5->SaveAs("effPur.root");
    //c5->SaveAs("effPur.pdf");
    c5->SaveAs("effPur.C");

    for (int eventType = 0; eventType<5; eventType++) {
        for(int outputType = 0; outputType<9; outputType++) {
            hists[eventType][outputType]->Scale( 1/(hists[eventType][outputType]->GetEntries()) );
            hists[eventType][outputType]->Write();
        }
    }

    mainOutput->Close();

}
