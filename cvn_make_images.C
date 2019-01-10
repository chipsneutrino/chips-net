/*
    ROOT Macro to generate the "Image" files for use in the CHIPS CVN from the WCSim output
*/

#include <vector>
#include <fstream>

void cvn_make_images(const char* in_dir="", const char* out_name="", 
                    int label=-999, int PDG_code=-999, 
                    bool save_parameters=true, int images_to_make=10000) {

    // Other Options
    int num_files           = 200;      // Number of files in input directory
    int num_hits_cut        = 100;      // Cut to apply on the number of hits
    bool make_plots         = true;    // Shall we produce a plots file with monitoring histograms
    const int image_x_bins  = 32;       // Number of x bins in the images
    const int image_y_bins  = 32;       // Number of y bins in the images

	// Load the libraries we need...
	std::cout << "Loading libraries..." << std::endl;
    TString libWCSimRoot = TString::Format("%s%s",gSystem->Getenv("WCSIMHOME"), "/libWCSimRoot.so");
    gSystem->Load(libWCSimRoot.Data());

    // Open the output .txt file
    ofstream output_file;
    output_file.open(out_name);

    // Make the plots file histograms
    TH1F *channel_hit_hist = new TH1F("channel_hit_hist","channel_hit_hist",100,0,20);
    TH1F *channel_time_hist = new TH1F("channel_time_hist","channel_time_hist",100,0,200);
    TH1F *spatial_theta_hist = new TH1F("spatial_theta_hist","spatial_theta_hist",100,-TMath::Pi()/2,TMath::Pi()/2);
    TH1F *spatial_phi_hist = new TH1F("spatial_phi_hist","spatial_phi_hist",100,-TMath::Pi(),TMath::Pi()); 
    TH1F *lepton_theta_hist = new TH1F("lepton_theta_hist","lepton_theta_hist",100,-TMath::Pi()/2,TMath::Pi()/2);
    TH1F *lepton_phi_hist = new TH1F("lepton_phi_hist","lepton_phi_hist",100,-TMath::Pi(),TMath::Pi());   
    TH1F *beam_E_hist = new TH1F("beam_E_hist","beam_E_hist", 100, 0, 5000);
    TH1F *lepton_E_hist = new TH1F("lepton_E_hist","lepton_E_hist", 100, 0, 5000);        

    // Variables to keep track of images we have made/skipped and the pmt locations
    int num_images = 0;
    int num_skipped = 0;
    std::vector<WCSimRootPMT> pmts;

    // Run through the input files and fill output .txt file
    char* dir = gSystem->ExpandPathName(in_dir);
    void* dirp = gSystem->OpenDirectory(dir);
    const char* entry;
    TString str;
    int n=0;
    while((entry = (char*)gSystem->GetDirEntry(dirp)) && n<num_files){
        str = entry;
        if(str.EndsWith(".root")){
            n++;
            std::cout << "Processing File [" << n << "] -> " << str << " ..." << std::endl;
            TFile * input_file = new TFile(gSystem->ConcatFileName(dir, entry), "READ");

            if(!input_file->GetListOfKeys()->Contains("wcsimT")){ 
                std::cout << "Skipping File" << std::endl; 
                input_file->Close();
                delete input_file;
                continue;
            }

            // Load the main TTree
            TTree *main_tree = (TTree*)input_file->Get("wcsimT");
            int num_events = main_tree->GetEntries();
            WCSimRootEvent* wcsimrootsuperevent = new WCSimRootEvent();
            TBranch *branch = main_tree->GetBranch("wcsimrootevent");

            if (branch == 0) {
                std::cout << "Skipping File" << std::endl; 
                input_file->Close();
                delete input_file;
                continue;
			}

            branch->SetAddress(&wcsimrootsuperevent);
            main_tree->GetBranch("wcsimrootevent")->SetAutoDelete(kTRUE);
            WCSimRootTrigger* wcsimrootevent;

            // If first file load the PMT info into the "pmts" variables
            if (n == 1) {
                // Get the geometry TTree
                TTree *gtree = (TTree*)input_file->Get("wcsimGeoT");
                WCSimRootGeom* wcsimrootgeom = new WCSimRootGeom();
                TBranch *geo_branch = gtree->GetBranch("wcsimrootgeom");
                geo_branch->SetAddress(&wcsimrootgeom);
                gtree->GetEntry(0);  

                // Run through all the PMTs and add to "pmts" vector
                for (int p=0; p < wcsimrootgeom->GetWCNumPMT(); p++) {
                    pmts.push_back(wcsimrootgeom->GetPMTFromArray(p));
                }              
            }

            // RUn through all the events in the file
            for(int evt=0; evt<num_events; evt++){
            	main_tree->GetEntry(evt);
                try { wcsimrootevent = wcsimrootsuperevent->GetTrigger(0);
                } catch (std::exception& e) {
                    std::cerr << "Exception catched : " << e.what() << std::endl;
                    num_skipped ++;
                    continue;
                }

                int num_cherenkov_digi_hits = wcsimrootevent->GetNcherenkovdigihits();
                if (num_cherenkov_digi_hits < num_hits_cut) { // THIS IS THE CUT!!!
                    num_skipped ++;
                    wcsimrootsuperevent->ReInitialize();
                    continue;
                }

                // Load the truthSummary
            	WCSimTruthSummary truth_summary = wcsimrootsuperevent->GetTruthSummary();
                float beamE = truth_summary.GetBeamEnergy();
                float vtxX, vtxY, vtxZ, vtxT, dirTheta, dirPhi, lepE;
                if (save_parameters) {
                    vtxX = truth_summary.GetVertexX(); 
                    vtxY = truth_summary.GetVertexY(); 
                    vtxZ = truth_summary.GetVertexZ(); 
                    vtxT = truth_summary.GetVertexT();
                    for (int p=0; p<truth_summary.GetNPrimaries(); p++) {
                        if (truth_summary.GetPrimaryPDG(p) == PDG_code) {
                            TVector3 leptonDir = truth_summary.GetPrimaryDir(p);
                            float dirX = leptonDir.X();
                            float dirY = leptonDir.Y();
                            float dirZ = leptonDir.Z();

                            if (dirX > 0 && dirY < 0)      { dirPhi = TMath::ATan(dirY/dirX); }
                            else if (dirX < 0 && dirY < 0) { dirPhi = TMath::ATan(dirY/dirX) - TMath::Pi(); }
                            else if (dirX < 0 && dirY > 0) { dirPhi = TMath::ATan(dirY/dirX) + TMath::Pi(); }
                            else if (dirX > 0 && dirY > 0) { dirPhi = TMath::ATan(dirY/dirX); }  
                            else { std::cout << "Error: Can't find barrel phi dir angle!" << std::endl; }

                            dirTheta = TMath::ATan(dirZ/(sqrt(pow(dirX,2) + pow(dirY,2))));

                            if (make_plots) {
                                lepton_theta_hist->Fill(dirTheta);
                                lepton_phi_hist->Fill(dirPhi);
                            }

                            lepE = truth_summary.GetPrimaryEnergy(p);
                        }
                    }
                }

                // Make hit hit and time histograms and initialise a large first_hit_time
                TH2F *hist_hit = new TH2F("hist_hit","hist_hit",image_x_bins,-TMath::Pi(),TMath::Pi(),image_y_bins,-TMath::Pi()/2,TMath::Pi()/2);
                TH2F *hist_time = new TH2F("hist_time","hist_time",image_x_bins,-TMath::Pi(),TMath::Pi(),image_y_bins,-TMath::Pi()/2,TMath::Pi()/2);              
                float first_hit_time = 1000000;

                // Loop through the digi hits...
                for(int h=0; h<num_cherenkov_digi_hits; h++){
                    TObject *element = (wcsimrootevent->GetCherenkovDigiHits())->At(h);
                    WCSimRootCherenkovDigiHit *wcsimrootcherenkovdigihit = dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
                    
                    int pmt_index = wcsimrootcherenkovdigihit->GetTubeId();
                    float hit_x = pmts[pmt_index].GetPosition(0);
                    float hit_y = pmts[pmt_index].GetPosition(1);
                    float hit_z = pmts[pmt_index].GetPosition(2);

                    float digi_hit_time = wcsimrootcherenkovdigihit->GetT();
                    if (digi_hit_time < first_hit_time) { first_hit_time = digi_hit_time; }

                    float digi_hit_q = wcsimrootcherenkovdigihit->GetQ();
            
                    if (hit_x == 0.0) {continue;} // Due to the divide in the ATan() method below
                    if (hit_z == 0.0) {continue;} // Due to the divide in the ATan() method below

                    float hit_phi = 0;
                    if (hit_x > 0 && hit_y < 0)      { hit_phi = TMath::ATan(hit_y/hit_x); }
                    else if (hit_x < 0 && hit_y < 0) { hit_phi = TMath::ATan(hit_y/hit_x) - TMath::Pi(); }
                    else if (hit_x < 0 && hit_y > 0) { hit_phi = TMath::ATan(hit_y/hit_x) + TMath::Pi(); }
                    else if (hit_x > 0 && hit_y > 0) { hit_phi = TMath::ATan(hit_y/hit_x); }  
                    else { std::cout << "Error: Can't find barrel phi angle!" << std::endl; }

                    float hit_theta = TMath::ATan(hit_z/(sqrt(pow(hit_x,2) + pow(hit_y,2))));

                    hist_hit->Fill(hit_phi, hit_theta, digi_hit_q);

                    if (make_plots) {
                        spatial_theta_hist->Fill(hit_theta, digi_hit_q);
                        spatial_phi_hist->Fill(hit_phi, digi_hit_q);
                    }

                    int bin_num = hist_time->FindBin(hit_phi, hit_theta);
                    float currentBinTime = hist_time->GetBinContent(bin_num);
                    if ((currentBinTime == 0) || (currentBinTime != 0 && digi_hit_time<currentBinTime)) {
                        hist_time->SetBinContent(bin_num, digi_hit_time);
                    }
                }   

                // Add the label and energy to the start of the line
                output_file << label << " ";
                output_file << beamE << " ";
                
                if (save_parameters) {
                    output_file << vtxX << " ";
                    output_file << vtxY << " ";
                    output_file << vtxZ << " ";
                    output_file << (vtxT-first_hit_time) << " ";
                    output_file << dirTheta << " ";
                    output_file << dirPhi << " ";
                    output_file << lepE << " ";    
                } else {
                    output_file << 0.0 << " ";
                    output_file << 0.0 << " ";
                    output_file << 0.0 << " ";
                    output_file << 0.0 << " ";
                    output_file << 0.0 << " ";
                    output_file << 0.0 << " ";
                    output_file << 0.0 << " ";                        
                }

                if (make_plots) {
                    beam_E_hist->Fill(beamE);
                    lepton_E_hist->Fill(lepE);                    
                }

                // Add the hit image to the line
                for (int x=1; x<=image_x_bins; x++) {
                    for (int y=1; y<=image_y_bins; y++) {
                        float content = hist_hit->GetBinContent(x,y);
                        if (content != 0 && make_plots) { channel_hit_hist->Fill(content); }
                        output_file << content << " ";
                    }
                }

                // Add the time image to the line. I PROBABLY WANT TO MAKE THIS AN INTEGER THING!!!
                for (int x=1; x<=image_x_bins; x++) {
                    for (int y=1; y<=image_y_bins; y++) {
                        float binContent = hist_time->GetBinContent(x, y);
                        if (binContent == 0) {
                            output_file << 0.0 << " ";    
                        } else {
                            if (make_plots) { channel_time_hist->Fill(binContent-first_hit_time); }
                            output_file << (binContent-first_hit_time) << " ";
                        }
                    }
                }

                // End the line
                output_file << "\n";

                num_images ++;
                delete hist_hit;
                delete hist_time;

                if (num_images >= images_to_make) { break; }

                wcsimrootsuperevent->ReInitialize();
            }
 
            delete wcsimrootevent;

            input_file->Close();
            delete input_file;

        } 

        if (num_images >= images_to_make) { break; }

    }

    if (make_plots) {
        TFile * plotFile = new TFile("channelPlots.root", "RECREATE");
        channel_hit_hist->Write();
        channel_time_hist->Write();
        spatial_theta_hist->Write();
        spatial_phi_hist->Write();
        lepton_theta_hist->Write();
        lepton_phi_hist->Write();
        beam_E_hist->Write();
        lepton_E_hist->Write();
        plotFile->Close();
    }

    // Close the output .txt file
    output_file.close();
    std::cout << "Num Skipped -> " << num_skipped << std::endl;
}
