# Polar-RepDETR Pseudocode

## Algorithm 1. Polar-RepDETR Training

**Input:** Original annotated datasets; model parameters $\theta$; the shared running mean $\bar L=1$; SGD optimizer; number of epochs $E$; training random seed $s=0$.

**Output:** The checkpoint selected by clean-validation fitness.

1. Split original images before augmentation. Prepare the bearing training set using 255 source images per class and two noise levels ($\sigma=20,40$); leave clean validation/test images unchanged. Use the paper's class-stratified NEU-DET split without offline augmentation.
2. For each epoch $e=1,\ldots,E$ do:
3. For each training mini-batch $(I,G)$ do:
4. Apply training-only online transforms and resize images to $640\times640$ (bearing) or $416\times416$ (NEU-DET); express target boxes as normalized center-xywh coordinates.
5. Extract backbone features $P_3,P_4,P_5$ using Conv, ADown and RepNCSPELAN4_CA (Algorithm 2).
6. Set $E_5=\operatorname{EPFI}(\operatorname{Proj}_5(P_5))$ (Algorithm 3), where each $\operatorname{Proj}$ is a $1\times1$ channel projection to 256 channels.
7. Set $Y_5=\operatorname{LDConv}_{1,1}(E_5)$ (Algorithm 4). The subscripts denote sampling-point count and stride, respectively.
8. Set $T_4=\operatorname{RepNCSPELAN4\_CA}([\operatorname{Up}_2(Y_5);\operatorname{Proj}_4(P_4)])$ and $Y_4=\operatorname{LDConv}_{1,1}(T_4)$.
9. Set $F_3=\operatorname{RepNCSPELAN4\_CA}([\operatorname{Up}_2(Y_4);\operatorname{Proj}_3(P_3)])$.
10. Set $F_4=\operatorname{RepNCSPELAN4\_CA}([\operatorname{LDConv}_{3,2}(F_3);Y_4])$ and $F_5=\operatorname{RepNCSPELAN4\_CA}([\operatorname{LDConv}_{3,2}(F_4);Y_5])$.
11. Decode $(F_3,F_4,F_5)$ to obtain encoder predictions, decoder-layer predictions and denoising predictions.
12. Form the ordered supervision list: final decoder output; encoder output; earlier decoder outputs in increasing depth; final denoising output; earlier denoising outputs in increasing depth. Omit denoising entries when unavailable. Set $L_{\rm total}=0$.
13. For each supervised prediction set $S$ in this list do:
14. Use the original Hungarian class/L1/GIoU matching cost for regular predictions; use known positive assignments for denoising predictions.
15. Compute the original RT-DETR classification loss $L_{\rm cls}^{S}$, including unmatched-query classification targets. For the $M$ matched pairs, set $L_1^{S}=M^{-1}\sum_{i=1}^{M}\|P_i-G_i\|_1$; use zero when $M=0$.
16. For $M>0$, call Algorithm 5 with the matched pairs and current $\bar L$ to obtain $L_{\rm reg}^{S}$ and updated $\bar L$; otherwise set $L_{\rm reg}^{S}=0$ and retain $\bar L$.
17. Accumulate $L_{\rm total}\gets L_{\rm total}+L_{\rm cls}^{S}+5L_1^{S}+2L_{\rm reg}^{S}$.
18. End for (supervised prediction sets).
19. Backpropagate $L_{\rm total}$ with the framework's gradient-accumulation scaling. At a scheduled optimizer step, update $\theta$ using SGD, update exponential-moving-average model weights and reset accumulated gradients.
20. End for (training mini-batches).
21. Evaluate the framework-selected evaluation weights on the clean validation set with running-mean updates disabled; save the checkpoint when validation fitness improves.
22. End for (epochs).
23. Return the selected checkpoint. Use clean and corrupted held-out test sets only for final evaluation.

## Algorithm 2. RepNCSPELAN4_CA

**Input:** Feature map $X\in\mathbb R^{B\times C_{\rm in}\times H\times W}$; even intermediate width $c_3$; branch width $c_4$; output width $C_{\rm out}$; learned module weights.

**Output:** Feature map $Y\in\mathbb R^{B\times C_{\rm out}\times H\times W}$.

1. Compute $U=\operatorname{Conv}_{1\times1}(X)$ with $c_3$ channels and split $U$ equally along channels into $(U_1,U_2)$. Here Conv denotes convolution, batch normalization and SiLU unless stated otherwise.
2. Compute $U_3=\operatorname{Conv}_{3\times3}(\operatorname{RepNCSP}(U_2))$ with $c_4$ channels.
3. Compute $U_4=\operatorname{Conv}_{3\times3}(\operatorname{RepNCSP}(U_3))$ with $c_4$ channels.
4. Concatenate along channels: $F=[U_1;U_2;U_3;U_4]\in\mathbb R^{B\times C\times H\times W}$, where $C=c_3+2c_4$ (Fig. 3a).
5. Pool along width and height: $z_c^h(h)=W^{-1}\sum_{j=0}^{W-1}F_c(h,j)$ and $z_c^w(w)=H^{-1}\sum_{j=0}^{H-1}F_c(j,w)$ (Eqs. 1--2; batch index omitted).
6. Transpose the width descriptor to align its spatial axis with the height descriptor; concatenate along that spatial axis to obtain a descriptor of shape $B\times C\times(H+W)\times1$.
7. Apply a shared $1\times1$ convolution to $m$ channels, batch normalization and h-swish to obtain $f$ (Eq. 3). Use $m=\max(8,\lfloor C/32\rfloor)$.
8. Split $f$ into lengths $H$ and $W$, transpose the width part back, and apply independent $1\times1$ convolutions to $C$ channels followed by sigmoid, yielding $a^h$ and $a^w$ (Eqs. 4--5).
9. Reweight $F'_c(h,w)=F_c(h,w)a_c^h(h)a_c^w(w)$ (Eq. 6).
10. Return $Y=\operatorname{Conv}_{1\times1}(F')$ with $C_{\rm out}$ channels.

## Algorithm 3. EPFI with POLA and EDFFN

**Input:** High-level feature map $X\in\mathbb R^{B\times C\times H\times W}$; number of heads $h_a=8$; learned POLA, EDFFN and normalization weights. Require $C$ divisible by $h_a$ and even head width $d=C/h_a$.

**Output:** Enhanced feature map $Y$ with the same shape as $X$.

1. Set the token count $T=HW$ and flatten $X$ to shape $B\times T\times C$. Project tokens to query $Q$, gate $G$, key $K$ and value $V$.
2. Interpolate the learned spatial encoding to $(H,W)$ when necessary and add it to both $Q$ and $K$. Reshape $Q,K,V$ to $B\times h_a\times T\times d$.
3. Divide $Q,K$ by the positive learned scale $\operatorname{Softplus}(S_0)$. Set the learned element-wise exponent $p=1+4\operatorname{sigmoid}(\Pi)$.
4. Set $Q^+=\operatorname{ReLU}(Q)^p$, $Q^-=\operatorname{ReLU}(-Q)^p$, $K^+=\operatorname{ReLU}(K)^p$ and $K^-=\operatorname{ReLU}(-K)^p$ (Eqs. 7--10).
5. Set $Q_s=[Q^+;Q^-]$, $Q_o=[Q^-;Q^+]$ and $K_s=[K^+;K^-]$ along the final feature dimension. Split $V=[V_s;V_o]$ into equal final-dimension halves.
6. Compute $M_s=T^{-1}K_s^\top V_s$, $M_o=T^{-1}K_s^\top V_o$ and $\bar K=T^{-1}\sum_{j=1}^{T}K_{s,j}$. All matrix products are evaluated independently per batch and head.
7. Compute $A_s=(Q_sM_s)\oslash(Q_s\bar K^\top+\epsilon)$ and $A_o=(Q_oM_o)\oslash(Q_o\bar K^\top+\epsilon)$, with $\epsilon=10^{-6}$. Each scalar denominator is broadcast over its corresponding value channels.
8. Concatenate $[A_s;A_o]$ within each head and merge heads. Add the depthwise $5\times5$ value-convolution branch, multiply element-wise by $G$, and apply the learned output projection.
9. Reshape the attention result to $A\in\mathbb R^{B\times C\times H\times W}$ and compute $Z=\operatorname{LayerNorm}(X+A)$ along channels.
10. For EDFFN, apply a $1\times1$ convolution from $C$ to $4C$ channels and a depthwise $3\times3$ convolution; split into $Z_1,Z_2$, each with $2C$ channels.
11. Set $U=\operatorname{Conv}_{1\times1}(\operatorname{GELU}(Z_1)\odot Z_2)$, projecting $2C$ channels back to $C$ before frequency filtering.
12. Pad the right and bottom to multiples of 8 and partition $U$ into $8\times8$ patches. Use reflection padding, with replication fallback when reflection is undefined for a very small spatial dimension.
13. Apply two-dimensional real FFT independently to each channel and patch. Multiply by the learned real weights $W_f\in\mathbb R^{C\times8\times5}$, shared across patches; then apply inverse real FFT with output size $8\times8$.
14. Reassemble patches and crop to $(H,W)$ to obtain $U_f$. Return $Y=\operatorname{LayerNorm}(Z+U_f)$.

## Algorithm 4. Linear Deformable Convolution

**Input:** Feature map $X\in\mathbb R^{B\times C_{\rm in}\times H\times W}$; integer sampling-point count $N_s\ge1$; positive integer stride $s$; offset and aggregation weights.

**Output:** Feature map $Y\in\mathbb R^{B\times C_{\rm out}\times H'\times W'}$, where $H'=\lceil H/s\rceil$ and $W'=\lceil W/s\rceil$.

1. Predict offsets $\Delta P=\operatorname{Conv}_{3\times3,s,p=1}(X)$ with $2N_s$ output channels. Initialize offset weights and biases to zero.
2. Set $c=\max(1,\operatorname{round}(\sqrt{N_s}))$ and construct $p_n=(\lfloor n/c\rfloor,n\bmod c)$ for $n=0,\ldots,N_s-1$. Coordinate pairs are ordered as (row, column).
3. For each output position $(h,w)$ and sampling index $n$, read the row offset from channel $n$ and the column offset from channel $N_s+n$.
4. Set $p_0=(sh,sw)$ and $p=p_0+p_n+\Delta p_n(h,w)$.
5. Clamp $p_y$ to $[0,H-1]$ and $p_x$ to $[0,W-1]$. Let $\mathcal N(p)$ be the unique in-bounds floor/ceiling corner coordinates surrounding $p$.
6. Interpolate $\widetilde X_n(h,w)=\sum_{q\in\mathcal N(p)}\max(0,1-|p_y-q_y|)\max(0,1-|p_x-q_x|)X(q)$. Each unique corner is counted once, including at image borders.
7. Stack samples in the order $S[:, :,hN_s+n,w]=\widetilde X_n(h,w)$, giving shape $B\times C_{\rm in}\times(H'N_s)\times W'$.
8. Apply an aggregation convolution with kernel $(N_s,1)$ and stride $(N_s,1)$, followed by batch normalization and SiLU; return $Y$.

## Algorithm 5. Wise-Inner-MPDIoU Loss Computation

**Input:** $P=\{(x_i,y_i,w_i,h_i)\}_{i=1}^N$: predicted boxes in xywh format; $G=\{(x_i^{gt},y_i^{gt},w_i^{gt},h_i^{gt})\}_{i=1}^N$: matched ground-truth boxes; $\bar L_{\rm prev}$: previous running mean; $\rho=0.7$, $\alpha=1.7$, $\delta=2.7$, $\mu=0.01$; $W_{\rm img},H_{\rm img}$: dimensions of the box-regression coordinate domain.

**Output:** $L$: Wise-Inner-MPDIoU loss; $\bar L$: updated running mean.

1. For $i=1$ to $N$ do:
2. Convert $P_i$ and $G_i$ from xywh format to corner coordinates.
3. Compute $W_{g,i},H_{g,i}$ as the width and height of the smallest enclosing box.
4. $R_{{\rm WIoU},i}\gets\exp\left(((x_i-x_i^{gt})^2+(y_i-y_i^{gt})^2)/(W_{g,i}^2+H_{g,i}^2)\right)$.
5. $b_{l,i}\gets x_i-w_i\rho/2$; $b_{r,i}\gets x_i+w_i\rho/2$.
6. $b_{t,i}\gets y_i-h_i\rho/2$; $b_{b,i}\gets y_i+h_i\rho/2$.
7. $b_{l,i}^{gt}\gets x_i^{gt}-w_i^{gt}\rho/2$; $b_{r,i}^{gt}\gets x_i^{gt}+w_i^{gt}\rho/2$.
8. $b_{t,i}^{gt}\gets y_i^{gt}-h_i^{gt}\rho/2$; $b_{b,i}^{gt}\gets y_i^{gt}+h_i^{gt}\rho/2$.
9. $\mathrm{inter}_i\gets\max(\min(b_{r,i},b_{r,i}^{gt})-\max(b_{l,i},b_{l,i}^{gt}),0)\times\max(\min(b_{b,i},b_{b,i}^{gt})-\max(b_{t,i},b_{t,i}^{gt}),0)$.
10. $\mathrm{union}_i\gets(w_i^{gt}h_i^{gt}+w_ih_i)\rho^2-\mathrm{inter}_i$.
11. $\mathrm{IoU}_{{\rm inner},i}\gets\mathrm{inter}_i/\mathrm{union}_i$.
12. $L_{{\rm inner},i}\gets1-\mathrm{IoU}_{{\rm inner},i}$.
13. $D_{{\rm MPDIoU},i}\gets\left(\|p_{1,i}^{pred}-p_{1,i}^{gt}\|_2^2+\|p_{2,i}^{pred}-p_{2,i}^{gt}\|_2^2\right)/(W_{\rm img}^2+H_{\rm img}^2)$.
14. End for.
15. $\bar L\gets(1-\mu)\bar L_{\rm prev}+\mu\,\operatorname{Mean}(\operatorname{StopGradient}(\{L_{{\rm inner},i}\}_{i=1}^N))$.
16. For $i=1$ to $N$ do:
17. $\beta_i\gets\operatorname{StopGradient}(L_{{\rm inner},i})/\bar L$.
18. $r_i\gets\beta_i/[\delta\alpha^{\beta_i-\delta}]$.
19. $L_i\gets r_iR_{{\rm WIoU},i}(L_{{\rm inner},i}+D_{{\rm MPDIoU},i})$.
20. End for.
21. $L\gets\operatorname{Mean}(\{L_i\}_{i=1}^N)$.
22. Return $L,\bar L$.

## Conventions

- Algorithm 5 requires $N\ge1$ and finite center-xywh boxes with positive widths/heights in a common coordinate domain. In Step 13, $p_1=(x-w/2,y-h/2)$ and $p_2=(x+w/2,y+h/2)$ are the original-box corners.
- For normalized boxes, use $W_{\rm img}=H_{\rm img}=1$. Clamp denominators to a positive numerical lower bound; retain gradients through the distance-attention denominator.
- Empty matched sets return zero without updating $\bar L$. Update $\bar L$ once per nonempty training loss call, in Algorithm 1 order; freeze it during evaluation.
