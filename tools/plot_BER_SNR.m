SNR = 5:5:40;
SNR_64_1 = 0:5:25;
SNR_64_2 = 0:5:25;
semilogy(SNR,BER_CE_OAMP_16QAM_match,'r-v');hold on
semilogy(SNR,BER_DL_OAMP_16QAM_mismatch_new_25dB,'b-o');
semilogy(SNR,BER_DL_OAMP_16QAM_match,'m--');
semilogy(SNR,BER_CE_OAMP_64QAM_match,'r-v');
semilogy(SNR,BER_DL_OAMP_64QAM_mismatch30dB,'b-o');
semilogy(SNR,BER_DL_OAMP_64QAM_mismatch35dB_T_5,'m--');
% semilogy(SNR,BER_OAMP_Net_4,'k:o');
% semilogy(SNR,BER_OAMP_Net_8,'k:s');
% semilogy(SNR_64_2,BER_OAMP_Net_32,'k:v');
%semilogy(SNR,BER_OAMP_Net2_8,'b:s');
xlabel('SNR(dB)'),ylabel('BER');
%legend('OAMP(M=4)','OAMP(M=8)','OAMP(M=32)','OAMP-Net(M=4)','OAMP-Net(M=8)','OAMP-Net(M=32)')
legend('CE\_OAMP','DL\_OAMP(mismatch 25dB)','DL\_OAMP(match)')
% title('64QAM£¨M:ÌìÏßÊý£©')
axis([5 40 1e-4 1]);
grid on;
