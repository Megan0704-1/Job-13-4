#id2 = affine_map<(d0,d1)->(d0,d1)>
#id3 = affine_map<(d0,d1,d2)->(d0,d1,d2)>

module {
  func.func @flash_attention_like(
      %Q: tensor<?x?x?xf32>,   // [B,M,D]
      %K: tensor<?x?x?xf32>,   // [B,N,D]
      %V: tensor<?x?x?xf32>    // [B,N,Dv]
  ) -> tensor<?x?x?xf32> {
    // ---- dims
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %B  = tensor.dim %Q, %c0 : tensor<?x?x?xf32>
    %M  = tensor.dim %Q, %c1 : tensor<?x?x?xf32>
    %D  = tensor.dim %Q, %c2 : tensor<?x?x?xf32>
    %N  = tensor.dim %K, %c1 : tensor<?x?x?xf32>
    %Dv = tensor.dim %V, %c2 : tensor<?x?x?xf32>

    // ---- consts
    %zero_f = arith.constant 0.0 : f32
    %neg_big = arith.constant -3.40282347E+38 : f32
    %one_f  = arith.constant 1.0 : f32

    // scale = 1/sqrt(D)
    %D_i64 = arith.index_cast %D : index to i64
    %D_f32 = arith.sitofp %D_i64 : i64 to f32
    %sqrtD = math.sqrt %D_f32 : f32
    %scale = arith.divf %one_f, %sqrtD : f32

    // ---- running stats m, l, O
    %m_e = tensor.empty(%B, %M) : tensor<?x?xf32>
    %m0  = linalg.fill ins(%neg_big : f32) outs(%m_e : tensor<?x?xf32>) -> tensor<?x?xf32>
    %l_e = tensor.empty(%B, %M) : tensor<?x?xf32>
    %l0  = linalg.fill ins(%zero_f : f32) outs(%l_e : tensor<?x?xf32>) -> tensor<?x?xf32>
    %O_e = tensor.empty(%B, %M, %Dv) : tensor<?x?x?xf32>
    %O0  = linalg.fill ins(%zero_f : f32) outs(%O_e : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // ---- tile size (assume N % TILE == 0 for simplicity)
    %TILE = arith.constant 64 : index

    // ---- stream over N in tiles
    %O_fin, %m_fin, %l_fin =
      scf.for %n0 = %c0 to %N step %TILE
        iter_args(%O_acc = %O0, %m_acc = %m0, %l_acc = %l0)
        -> (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {

        // K_tile:[B,TILE,D], V_tile:[B,TILE,Dv]
        %K_tile = tensor.extract_slice %K[%c0, %n0, %c0] [%B, %TILE, %D] [1, 1, 1]
                  : tensor<?x?x?xf32> to tensor<?x?x?xf32>
        %V_tile = tensor.extract_slice %V[%c0, %n0, %c0] [%B, %TILE, %Dv] [1, 1, 1]
                  : tensor<?x?x?xf32> to tensor<?x?x?xf32>

        // S[b,m,t] = sum_k Q[b,m,k] * K_tile[b,t,k]
        %S_e  = tensor.empty(%B, %M, %TILE) : tensor<?x?x?xf32>
        %S_z  = linalg.fill ins(%zero_f : f32) outs(%S_e : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %S = linalg.generic
          { indexing_maps = [
              affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>,  // Q[b,m,k]
              affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>,  // K[b,t,k]
              affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>   // out[b,m,t]
            ],
            iterator_types = ["parallel","parallel","parallel","reduction"] }
          ins(%Q, %K_tile : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
          outs(%S_z : tensor<?x?x?xf32>) {
            ^bb0(%q: f32, %k: f32, %acc: f32):
              %mul = arith.mulf %q, %k : f32
              %sum = arith.addf %acc, %mul : f32
              linalg.yield %sum : f32
        } -> tensor<?x?x?xf32>

        // S_sc = S * scale
        %Ssc_e = tensor.empty(%B, %M, %TILE) : tensor<?x?x?xf32>
        %S_sc = linalg.generic
          { indexing_maps = [#id3, #id3],
            iterator_types = ["parallel","parallel","parallel"] }
          ins(%S : tensor<?x?x?xf32>) outs(%Ssc_e : tensor<?x?x?xf32>) {
            ^bb0(%x: f32, %out: f32):
              %y = arith.mulf %x, %scale : f32
              linalg.yield %y : f32
        } -> tensor<?x?x?xf32>

        // rmax_t[b,m] = max_t S_sc[b,m,t]
        %rmax_e = tensor.empty(%B, %M) : tensor<?x?xf32>
        %rmax_z = linalg.fill ins(%neg_big : f32) outs(%rmax_e : tensor<?x?xf32>) -> tensor<?x?xf32>
        %rmax_t = linalg.generic
          { indexing_maps = [
              affine_map<(d0,d1,d2)->(d0,d1,d2)>,
              affine_map<(d0,d1,d2)->(d0,d1)>
            ],
            iterator_types = ["parallel","parallel","reduction"] }
          ins(%S_sc : tensor<?x?x?xf32>) outs(%rmax_z : tensor<?x?xf32>) {
            ^bb0(%v: f32, %acc: f32):
              %gt  = arith.cmpf ogt, %v, %acc : f32
              %sel = arith.select %gt, %v, %acc : f32
              linalg.yield %sel : f32
        } -> tensor<?x?xf32>

        // m_new = max(m_acc, rmax_t)
        %mnew_e = tensor.empty(%B, %M) : tensor<?x?xf32>
        %m_new = linalg.generic
          { indexing_maps = [#id2, #id2, #id2],
            iterator_types = ["parallel","parallel"] }
          ins(%m_acc, %rmax_t : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%mnew_e : tensor<?x?xf32>) {
            ^bb0(%a: f32, %b: f32, %out: f32):
              %gt  = arith.cmpf ogt, %a, %b : f32
              %sel = arith.select %gt, %a, %b : f32
              linalg.yield %sel : f32
        } -> tensor<?x?xf32>

        // centered = S_sc - broadcast(m_new)
        %cent_e = tensor.empty(%B, %M, %TILE) : tensor<?x?x?xf32>
        %center = linalg.generic
          { indexing_maps = [
              affine_map<(d0,d1,d2)->(d0,d1,d2)>, // S_sc
              affine_map<(d0,d1,d2)->(d0,d1)>,    // m_new (broadcast)
              affine_map<(d0,d1,d2)->(d0,d1,d2)>  // out
            ],
            iterator_types = ["parallel","parallel","parallel"] }
          ins(%S_sc, %m_new : tensor<?x?x?xf32>, tensor<?x?xf32>)
          outs(%cent_e : tensor<?x?x?xf32>) {
            ^bb0(%s: f32, %m_: f32, %out: f32):
              %diff = arith.subf %s, %m_ : f32
              linalg.yield %diff : f32
        } -> tensor<?x?x?xf32>

        // E_new = exp(center)
        %E_e = tensor.empty(%B, %M, %TILE) : tensor<?x?x?xf32>
        %E_new = linalg.generic
          { indexing_maps = [#id3, #id3],
            iterator_types = ["parallel","parallel","parallel"] }
          ins(%center : tensor<?x?x?xf32>) outs(%E_e : tensor<?x?x?xf32>) {
            ^bb0(%x: f32, %out: f32):
              %e = math.exp %x : f32
              linalg.yield %e : f32
        } -> tensor<?x?x?xf32>

        // alpha = exp(m_acc - m_new)
        %alpha_e = tensor.empty(%B, %M) : tensor<?x?xf32>
        %alpha = linalg.generic
          { indexing_maps = [#id2, #id2, #id2],
            iterator_types = ["parallel","parallel"] }
          ins(%m_acc, %m_new : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%alpha_e : tensor<?x?xf32>) {
            ^bb0(%ma: f32, %mn: f32, %out: f32):
              %d = arith.subf %ma, %mn : f32
              %e = math.exp %d : f32
              linalg.yield %e : f32
        } -> tensor<?x?xf32>

        // l_t = sum_t E_new
        %lt_e0 = tensor.empty(%B, %M) : tensor<?x?xf32>
        %lt_z  = linalg.fill ins(%zero_f : f32) outs(%lt_e0 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %l_t = linalg.generic
          { indexing_maps = [
              affine_map<(d0,d1,d2)->(d0,d1,d2)>,
              affine_map<(d0,d1,d2)->(d0,d1)>
            ],
            iterator_types = ["parallel","parallel","reduction"] }
          ins(%E_new : tensor<?x?x?xf32>) outs(%lt_z : tensor<?x?xf32>) {
            ^bb0(%v: f32, %acc: f32):
              %s = arith.addf %acc, %v : f32
              linalg.yield %s : f32
        } -> tensor<?x?xf32>

        // l_new = alpha*l_acc + l_t
        %al_e = tensor.empty(%B, %M) : tensor<?x?xf32>
        %al = linalg.generic
          { indexing_maps = [#id2, #id2, #id2],
            iterator_types = ["parallel","parallel"] }
          ins(%alpha, %l_acc : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%al_e : tensor<?x?xf32>) {
            ^bb0(%a: f32, %l_: f32, %out: f32):
              %p = arith.mulf %a, %l_ : f32
              linalg.yield %p : f32
        } -> tensor<?x?xf32>

        %lnew_e = tensor.empty(%B, %M) : tensor<?x?xf32>
        %l_new = linalg.generic
          { indexing_maps = [#id2, #id2, #id2],
            iterator_types = ["parallel","parallel"] }
          ins(%al, %l_t : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%lnew_e : tensor<?x?xf32>) {
            ^bb0(%x: f32, %y: f32, %out: f32):
              %s = arith.addf %x, %y : f32
              linalg.yield %s : f32
        } -> tensor<?x?xf32>

        // SV = sum_t E_new[b,m,t] * V_tile[b,t,dv]  -> [B,M,Dv]
        %SV_e  = tensor.empty(%B, %M, %Dv) : tensor<?x?x?xf32>
        %SV_z  = linalg.fill ins(%zero_f : f32) outs(%SV_e : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %SV = linalg.generic
          { indexing_maps = [
              affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>,  // E_new[b,m,t]
              affine_map<(d0,d1,d2,d3)->(d0,d3,d2)>,  // V_tile[b,t,dv]
              affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>   // out[b,m,dv]
            ],
            iterator_types = ["parallel","parallel","parallel","reduction"] }
          ins(%E_new, %V_tile : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
          outs(%SV_z : tensor<?x?x?xf32>) {
            ^bb0(%a: f32, %b: f32, %acc: f32):
              %p = arith.mulf %a, %b : f32
              %s = arith.addf %acc, %p : f32
              linalg.yield %s : f32
        } -> tensor<?x?x?xf32>

        // Oscaled = (alpha*l_acc) broadcast-mul O_acc
        %al2_e = tensor.empty(%B, %M) : tensor<?x?xf32>
        %al2 = linalg.generic
          { indexing_maps = [#id2, #id2, #id2],
            iterator_types = ["parallel","parallel"] }
          ins(%alpha, %l_acc : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%al2_e : tensor<?x?xf32>) {
            ^bb0(%a: f32, %l_: f32, %out: f32):
              %p = arith.mulf %a, %l_ : f32
              linalg.yield %p : f32
        } -> tensor<?x?xf32>

        %Oscaled_e = tensor.empty(%B, %M, %Dv) : tensor<?x?x?xf32>
        %Oscaled = linalg.generic
          { indexing_maps = [
              affine_map<(d0,d1,d2)->(d0,d1)>,     // al2 broadcast
              affine_map<(d0,d1,d2)->(d0,d1,d2)>,  // O_acc
              affine_map<(d0,d1,d2)->(d0,d1,d2)>
            ],
            iterator_types = ["parallel","parallel","parallel"] }
          ins(%al2, %O_acc : tensor<?x?xf32>, tensor<?x?x?xf32>)
          outs(%Oscaled_e : tensor<?x?x?xf32>) {
            ^bb0(%w: f32, %o: f32, %out: f32):
              %p = arith.mulf %w, %o : f32
              linalg.yield %p : f32
        } -> tensor<?x?x?xf32>

        // Numer = Oscaled + SV
        %Num_e = tensor.empty(%B, %M, %Dv) : tensor<?x?x?xf32>
        %Numer = linalg.generic
          { indexing_maps = [#id3, #id3, #id3],
            iterator_types = ["parallel","parallel","parallel"] }
          ins(%Oscaled, %SV : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
          outs(%Num_e : tensor<?x?x?xf32>) {
            ^bb0(%a: f32, %b: f32, %out: f32):
              %s = arith.addf %a, %b : f32
              linalg.yield %s : f32
        } -> tensor<?x?x?xf32>

        // O_new = Numer / l_new  (broadcast)
        %Onew_e = tensor.empty(%B, %M, %Dv) : tensor<?x?x?xf32>
        %O_new = linalg.generic
          { indexing_maps = [
              #id3,                                  // Numer
              affine_map<(d0,d1,d2)->(d0,d1)>,      // l_new broadcast
              #id3
            ],
            iterator_types = ["parallel","parallel","parallel"] }
          ins(%Numer, %l_new : tensor<?x?x?xf32>, tensor<?x?xf32>)
          outs(%Onew_e : tensor<?x?x?xf32>) {
            ^bb0(%num: f32, %den: f32, %out: f32):
              %q = arith.divf %num, %den : f32
              linalg.yield %q : f32
        } -> tensor<?x?x?xf32>

        // ---- Tie results to iter_args via identity generics (OSB/SCF equivalence)
          %O_eq = tensor.insert_slice %O_new into %O_acc[%c0, %c0, %c0] [%B, %M, %Dv] [1, 1, 1]
  : tensor<?x?x?xf32> into tensor<?x?x?xf32>
%m_eq = tensor.insert_slice %m_new into %m_acc[%c0, %c0] [%B, %M] [1, 1]
  : tensor<?x?xf32> into tensor<?x?xf32>
%l_eq = tensor.insert_slice %l_new into %l_acc[%c0, %c0] [%B, %M] [1, 1]
  : tensor<?x?xf32> into tensor<?x?xf32>

scf.yield %O_eq, %m_eq, %l_eq
  : tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
      }

    return %O_fin : tensor<?x?x?xf32>
  }

  // --------------------- demo main ---------------------
  func.func @main() -> i32 {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    // Q init
    %Qe0 = tensor.empty() : tensor<4x256x512xf32>
    %Qz  = linalg.fill ins(%zero : f32) outs(%Qe0 : tensor<4x256x512xf32>) -> tensor<4x256x512xf32>
    %Qe1 = tensor.empty() : tensor<4x256x512xf32>
    %Q   = linalg.generic {indexing_maps = [#id3, #id3],
                           iterator_types = ["parallel","parallel","parallel"]}
           ins(%Qz : tensor<4x256x512xf32>) outs(%Qe1 : tensor<4x256x512xf32>) {
      ^bb0(%in: f32, %old: f32):
        %i0 = linalg.index 0 : index
        %i1 = linalg.index 1 : index
        %i2 = linalg.index 2 : index
        %a = arith.index_cast %i0 : index to i32
        %b = arith.index_cast %i1 : index to i32
        %c = arith.index_cast %i2 : index to i32
        %c7  = arith.constant 7  : i32
        %c3  = arith.constant 3  : i32
        %c13 = arith.constant 13 : i32
        %f13 = arith.constant 1.300000e+01 : f32
        %t0 = arith.muli %a, %c7  : i32
        %t1 = arith.muli %b, %c3  : i32
        %t2 = arith.addi %t0, %t1 : i32
        %t3 = arith.addi %t2, %c  : i32
        %t4 = arith.remsi %t3, %c13 : i32
        %f  = arith.sitofp %t4 : i32 to f32
        %v  = arith.divf %f, %f13 : f32
        linalg.yield %v : f32
    } -> tensor<4x256x512xf32>

    // K init
    %Ke0 = tensor.empty() : tensor<4x256x512xf32>
    %Kz  = linalg.fill ins(%zero : f32) outs(%Ke0 : tensor<4x256x512xf32>) -> tensor<4x256x512xf32>
    %Ke1 = tensor.empty() : tensor<4x256x512xf32>
    %K   = linalg.generic {indexing_maps = [#id3, #id3],
                           iterator_types = ["parallel","parallel","parallel"]}
           ins(%Kz : tensor<4x256x512xf32>) outs(%Ke1 : tensor<4x256x512xf32>) {
      ^bb0(%in: f32, %old: f32):
        %i0 = linalg.index 0 : index
        %i1 = linalg.index 1 : index
        %i2 = linalg.index 2 : index
        %a = arith.index_cast %i0 : index to i32
        %b = arith.index_cast %i1 : index to i32
        %c = arith.index_cast %i2 : index to i32
        %c11 = arith.constant 11 : i32
        %c5  = arith.constant 5  : i32
        %c17 = arith.constant 17 : i32
        %f17 = arith.constant 1.700000e+01 : f32
        %t0 = arith.muli %a, %c11 : i32
        %t1 = arith.muli %b, %c5  : i32
        %t2 = arith.addi %t0, %t1 : i32
        %t3 = arith.addi %t2, %c  : i32
        %t4 = arith.remsi %t3, %c17 : i32
        %f  = arith.sitofp %t4 : i32 to f32
        %v  = arith.divf %f, %f17 : f32
        linalg.yield %v : f32
    } -> tensor<4x256x512xf32>

    // V init
    %Ve0 = tensor.empty() : tensor<4x256x512xf32>
    %Vz  = linalg.fill ins(%zero : f32) outs(%Ve0 : tensor<4x256x512xf32>) -> tensor<4x256x512xf32>
    %Ve1 = tensor.empty() : tensor<4x256x512xf32>
    %V   = linalg.generic {indexing_maps = [#id3, #id3],
                           iterator_types = ["parallel","parallel","parallel"]}
           ins(%Vz : tensor<4x256x512xf32>) outs(%Ve1 : tensor<4x256x512xf32>) {
      ^bb0(%in: f32, %old: f32):
        %i0 = linalg.index 0 : index
        %i1 = linalg.index 1 : index
        %i2 = linalg.index 2 : index
        %a = arith.index_cast %i0 : index to i32
        %b = arith.index_cast %i1 : index to i32
        %c = arith.index_cast %i2 : index to i32
        %c13 = arith.constant 13 : i32
        %c7  = arith.constant 7  : i32
        %c19 = arith.constant 19 : i32
        %f19 = arith.constant 1.900000e+01 : f32
        %t0 = arith.muli %a, %c13 : i32
        %t1 = arith.muli %b, %c7  : i32
        %t2 = arith.addi %t0, %t1 : i32
        %t3 = arith.addi %t2, %c  : i32
        %t4 = arith.remsi %t3, %c19 : i32
        %f  = arith.sitofp %t4 : i32 to f32
        %v  = arith.divf %f, %f19 : f32
        linalg.yield %v : f32
    } -> tensor<4x256x512xf32>

    // Cast to dynamic for the callee signature
    %Qd = tensor.cast %Q : tensor<4x256x512xf32> to tensor<?x?x?xf32>
    %Kd = tensor.cast %K : tensor<4x256x512xf32> to tensor<?x?x?xf32>
    %Vd = tensor.cast %V : tensor<4x256x512xf32> to tensor<?x?x?xf32>

    // Run 100 iters; accumulate one element to avoid DCE and return an int.
    %acc0f = arith.constant 0.0 : f32
    %accf = scf.for %it = %c0 to %c100 step %c1 iter_args(%s = %acc0f) -> (f32) {
      %O = func.call @flash_attention_like(%Qd, %Kd, %Vd)
           : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %Ost = tensor.cast %O : tensor<?x?x?xf32> to tensor<4x256x512xf32>
      %v = tensor.extract %Ost[%c0, %c0, %c0] : tensor<4x256x512xf32>
      %s1 = arith.addf %s, %v : f32
      scf.yield %s1 : f32
    }
    %ret = arith.fptosi %accf : f32 to i32
    return %ret : i32
  }
}

