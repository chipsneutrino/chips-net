void cvn_parameter_output_plots(int par = 6, const char* outputFile = "output/parameter.txt", const char* plotFileName = "plots/parameter_plots.root") {

    // Open and read the network output file into a tree
    TTree *tree = new TTree("tree", "tree");
    tree->ReadFile(outputFile, "p1:p2:diff");

    // Open the plot file to write output plots to
    TFile * plotFile = new TFile(plotFileName,"RECREATE");

    // Create histograms
    TH1F * par_diff_hist;
    TH2F * par_2d_hist; 
    if (par==0) { // vtx x-position
        par_diff_hist = new TH1F("par_diff_hist", "vtxX_diff", 50, -2000, 2000); 
        par_diff_hist->GetXaxis()->SetTitle("True - Reco vtxX [cm]");
        par_2d_hist = new TH2F("par_2d_hist", "vtxX_2d", 50, -12500, 12500, 50, -12500, 12500);
    } else if (par==1) { // vtx y-position
        par_diff_hist = new TH1F("par_diff_hist", "vtxY_diff", 50, -2000, 2000); 
        par_diff_hist->GetXaxis()->SetTitle("True - Reco vtxY [cm]");
        par_2d_hist = new TH2F("par_2d_hist", "vtxY_2d", 50, -12500, 12500, 50, -12500, 12500);
    } else if (par==2) { // vtx z-position
        par_diff_hist = new TH1F("par_diff_hist", "vtxZ_diff", 50, -2000, 2000); 
        par_diff_hist->GetXaxis()->SetTitle("True - Reco vtxZ [cm]");
        par_2d_hist = new TH2F("par_2d_hist", "vtxZ_2d", 50, -12500, 12500, 50, -12500, 12500);
    } else if (par==3) { // vtx time
        par_diff_hist = new TH1F("par_diff_hist", "vtxT_diff", 50, -10, 10); 
        par_diff_hist->GetXaxis()->SetTitle("True - Reco vtxT [ns]");
        par_2d_hist = new TH2F("par_2d_hist", "vtxT_2d", 50, 0, 10000, 50, 0, 10000);
    } else if (par==4) { // track dir theta
        par_diff_hist = new TH1F("par_diff_hist", "theta_diff", 60, -1, 1); 
        par_diff_hist->GetXaxis()->SetTitle("True - Reco Track Theta Direction [radians]");
        par_2d_hist = new TH2F("par_2d_hist", "theta_2d", 50, -TMath::Pi()/2,TMath::Pi()/2, 50, -TMath::Pi()/2,TMath::Pi()/2);
    } else if (par==5) { // track dir phi
        par_diff_hist = new TH1F("par_diff_hist", "phi_diff", 60, -1, 1);
        par_diff_hist->GetXaxis()->SetTitle("True - Reco Track Phi Direction [radians]");
        par_2d_hist = new TH2F("par_2d_hist", "phi_2d", 50, -TMath::Pi(),TMath::Pi(), 50, -TMath::Pi(),TMath::Pi());
    } else if (par==6) { // track energy
        par_diff_hist = new TH1F("par_diff_hist", "energy_diff", 60, -2500, 2500); 
        par_diff_hist->GetXaxis()->SetTitle("True - Reco Theta Energy [MeV]");
        par_2d_hist = new TH2F("par_2d_hist", "energy_2d", 50, 0, 5000, 50, 0, 5000);
    }

    // Fill histograms
    tree->Draw("diff>>par_diff_hist");
    tree->Draw("p1:p2>>par_2d_hist");

    // Beautify main histogram
    par_diff_hist->GetYaxis()->SetTitle("Fraction of Events");
    par_diff_hist->GetXaxis()->CenterTitle();
    par_diff_hist->GetYaxis()->CenterTitle();
    par_diff_hist->Scale( 1/(par_diff_hist->GetEntries()) );
    
    par_diff_hist->Write();
    par_2d_hist->Write();
    plotFile->Close();
}