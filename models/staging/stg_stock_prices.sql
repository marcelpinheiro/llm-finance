with source as (
    select * from {{ source('finance', 'stock_prices') }}
),

renamed as (
    select
        id,
        company_id,
        date,
        open_price,
        close_price,
        high_price,
        low_price,
        volume,
        adjusted_close,
        created_at
    from source
)

select * from renamed
