function [ G, CF, P, W, H ] = AdaptiveSTFT( s, Type, CFtype, mode )
% AdaptiveSTFT creats adaptive STFT of the signal s using Instantaneous
% window width
% Written by Mohammad Bejvani
% Email--> mohammadbejvani@gmail.com and in Github--> @MBejvani

N = length(s);
s = s(:)';
s = s(1:2*fix(N/2));
l = fix(N/10);% N/u : u depend on signal
L = 2*l;
s = [zeros(1,l) s zeros(1,l)];
xidx = (1:L)';
yidx = 0:(N-1);
h = xidx(:,ones(N,1)) + yidx(ones(L,1),:);
H = s(h); % Henkel Matrix of the signal
switch Type
    case 'adaptive'
        %% Optimization Via Different Cost Functions
        alpha = [1:2:45];    % :2:
        for i = 1:length(alpha)
            w = gausswin(L,alpha(i));w = w/sum(w);
            wH = sparse(diag(w)) * H;
            TF = fft(wH);TF = TF(1:fix(L/2),:);
            TF = conj(TF).*TF;
            switch CFtype
                case 'log'
                    CF(i,:) = sum(log(TF + eps));
                case 'norm'
                    CF(i,:) = sum(TF)./max(TF,[],1);
                case 'exp'
                    CF(i,:) = sum(TF.^.2)./sum(exp(TF));
                case 'pow2'
                    CF(i,:) = sum(TF).^2./(sum(TF.^2)+1e-10);
            end
        end
        %% Windows Matrix
        if mode==1
            CF = sum(CF,2);
            [Q,P] = min(CF);
            w = gausswin(L,alpha(P));w = w./sum(w);W = repmat(w,1,N);
            P = [Q P];
        elseif mode==2
            [~,P] = min(CF);
            for i = 1:N
                W(:,i) = gausswin(L,alpha(P(i)));
            end
            SW = 1./sum(W);
            W  = W .* SW; % Normalization
        end
    case 'simple'
        w = gausswin(L,15);w = w./sum(w);W = repmat(w,1,N);
        CF=[]; P=[];
end
G = fft(W .* H,N);
G = abs(G(fix(N/2):end,:));
end

