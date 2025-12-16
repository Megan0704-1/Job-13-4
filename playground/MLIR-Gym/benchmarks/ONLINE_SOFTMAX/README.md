This is application is an implementation of online softmax kernel.

Algorithm:
- For each row `i`:
    - For each element `e`:
        1. m_i = max(m_old_i, e)
        2. s_i = s * exp(m_old_i - m_i) + exp(e - m_i)

-> output m_i, s_i per row

- Broadcast
- Do exp(X - m_i) / s_i

Why can we update scale like this?
- becuase denominator : exp(X - m_old) -> exp(X - m_new) * exp(m_new - m_old)
- and nominator: sum { exp(X - m_old) } -> exp(m_new-m_old) * sum { exp(X - m_new) }
